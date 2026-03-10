import torch
import torch.nn as nn

class MeanPoolingClassifier(nn.Module):
    """Pour l'expérience Patch par Patch (Single Scale)"""
    def __init__(self, num_vars, num_classes, in_features=384, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(num_vars * in_features, num_classes)

    def forward(self, x):
        B, S, F = x.shape
        P = S // self.num_vars
        
        # 1. Séparation et Moyenne sur la dimension des patchs
        x_reshaped = x.view(B, self.num_vars, P, F)
        pooled_vars = x_reshaped.mean(dim=2) 
        
        # 2. Flatten et Couche Linéaire
        final_repr = pooled_vars.view(B, -1)
        return self.linear(self.dropout(final_repr))

class MultiScaleMeanPoolingClassifier(nn.Module):
    def __init__(self, num_vars, num_classes, patches_counts, in_features=384, **kwargs):
        super().__init__()
        self.num_vars = num_vars
        self.p_counts = patches_counts
        self.dropout = nn.Dropout(0.1)
        
        self.linear = nn.Linear(num_vars * in_features * len(patches_counts), num_classes)

    def forward(self, x):
        B, _, F = x.shape
        start = 0
        pooled_list = []
        
        for scale in [64, 32, 16, 8]:
            length = self.num_vars * self.p_counts[scale]
            x_scale = x[:, start : start + length, :]
            start += length
            
            # 2. On isole et on fait la moyenne sur cette échelle
            x_scale_reshaped = x_scale.view(B, self.num_vars, self.p_counts[scale], F)
            pooled = x_scale_reshaped.mean(dim=2) 
            pooled_list.append(pooled.view(B, -1)) 
            
        # 3. Concaténation des moyennes et Couche Linéaire
        final_repr = torch.cat(pooled_list, dim=1) 
        return self.linear(self.dropout(final_repr))

# ==========================================
# 2. SINGLE SCALE ATTENTION
# ==========================================
class SingleScaleAttentionClassifier(nn.Module):
    def __init__(self, num_vars, num_classes, in_features=384, mode="shared_context"):
        super().__init__()
        self.num_vars = num_vars
        self.mode = mode

        if mode == "shared_context":
            self.context = nn.Parameter(torch.randn(in_features) * 0.1)
        elif mode == "independent_context":
            self.context = nn.Parameter(torch.randn(num_vars, in_features) * 0.1)
            
        self.attn_dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(num_vars * in_features, num_classes)

    def forward(self, x):
        B, S, F = x.shape
        P = S // self.num_vars
        x_reshaped = x.view(B, self.num_vars, P, F)
        
        if self.mode == "shared_context":
            scores = torch.matmul(x_reshaped, self.context) 
        elif self.mode == "independent_context":
            context_view = self.context.view(1, self.num_vars, 1, F)
            scores = (x_reshaped * context_view).sum(dim=-1) 
            
        weights = torch.softmax(scores, dim=2).unsqueeze(-1) 
        weights = self.attn_dropout(weights)
        pooled_vars = (weights * x_reshaped).sum(dim=2) 
        pooled = pooled_vars.view(B, -1) 
        return self.linear(self.dropout(pooled))

class SingleScaleMultiHeadClassifier(nn.Module):
    def __init__(self, num_vars, num_classes, in_features=384, num_heads=4, patch_mode="shared_context"):
        super().__init__()
        self.num_vars = num_vars
        self.mode = patch_mode
        
        if self.mode == "shared_context":
            self.cls_token = nn.Parameter(torch.randn(1, 1, in_features) * 0.02)
        elif self.mode == "independent_context":
            self.cls_token = nn.Parameter(torch.randn(num_vars, 1, in_features) * 0.02)

        self.mha = nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(num_vars * in_features, num_classes)

    def forward(self, x):
        B, S, F = x.shape
        P = S // self.num_vars
        x_reshaped = x.view(B, self.num_vars, P, F)
        kv = x_reshaped.view(B * self.num_vars, P, F)
        
        if self.mode == "shared_context":
            q = self.cls_token.expand(B * self.num_vars, -1, -1)
        elif self.mode == "independent_context":
            q_expanded = self.cls_token.unsqueeze(0).expand(B, -1, -1, -1)
            q = q_expanded.reshape(B * self.num_vars, 1, F)
            
        attn_output, _ = self.mha(query=q, key=kv, value=kv)
        pooled_flat = attn_output.squeeze(1) 
        pooled_vars = pooled_flat.view(B, self.num_vars, F) 
        pooled = pooled_vars.view(B, -1) 
        return self.linear(self.dropout(pooled))

class HierarchicalMultiHeadClassifier(nn.Module):
    def __init__(self, num_vars, num_classes, in_features=384, num_heads=4, patch_mode="independent_context"):
        super().__init__()
        self.num_vars = num_vars
        self.patch_mode = patch_mode
        
        if patch_mode == "shared_context":
            self.patch_cls = nn.Parameter(torch.randn(1, 1, in_features) * 0.02)
        elif patch_mode == "independent_context":
            self.patch_cls = nn.Parameter(torch.randn(num_vars, 1, in_features) * 0.02)

        self.patch_mha = nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.var_cls = nn.Parameter(torch.randn(1, 1, in_features) * 0.02)
        self.var_mha = nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        B, S, F = x.shape
        P = S // self.num_vars
        
        x_reshaped = x.view(B, self.num_vars, P, F)
        kv_patch = x_reshaped.view(B * self.num_vars, P, F)
        
        if self.patch_mode == "shared_context":
            q_patch = self.patch_cls.expand(B * self.num_vars, -1, -1)
        elif self.patch_mode == "independent_context":
            q_expanded = self.patch_cls.unsqueeze(0).expand(B, -1, -1, -1)
            q_patch = q_expanded.reshape(B * self.num_vars, 1, F)
            
        patch_attn_out, _ = self.patch_mha(query=q_patch, key=kv_patch, value=kv_patch)
        var_reprs = patch_attn_out.squeeze(1).view(B, self.num_vars, F) 
        
        kv_var = var_reprs 
        q_var = self.var_cls.expand(B, -1, -1) 
        var_attn_out, _ = self.var_mha(query=q_var, key=kv_var, value=kv_var)
        global_repr = var_attn_out.squeeze(1) 
        return self.linear(self.dropout(global_repr))

# ==========================================
# 3. CROSS-SCALE (MULTI-PATCH) ATTENTION
# ==========================================
class SequentialCrossScaleClassifier(nn.Module):
    def __init__(self, num_vars, num_classes, patches_counts, in_features=384, num_heads=8, shared_layer=False):
        super().__init__()
        self.num_vars = num_vars
        self.p_counts = patches_counts
        self.shared_layer = shared_layer
        
        if shared_layer:
            self.shared_attn = nn.MultiheadAttention(in_features, num_heads, dropout=0.1, batch_first=True)
        else:
            self.attn_32 = nn.MultiheadAttention(in_features, num_heads, dropout=0.1, batch_first=True)
            self.attn_16 = nn.MultiheadAttention(in_features, num_heads, dropout=0.1, batch_first=True)
            self.attn_8  = nn.MultiheadAttention(in_features, num_heads, dropout=0.1, batch_first=True)
        
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(num_vars * in_features, num_classes)

    def forward(self, x):
        B, _, F = x.shape
        start = 0
        x_dict = {}
        for scale in [64, 32, 16, 8]:
            length = self.num_vars * self.p_counts[scale]
            x_dict[scale] = x[:, start : start + length, :]
            start += length
            
        x_64_r = x_dict[64].reshape(B, self.num_vars, self.p_counts[64], F)
        seq_q = x_64_r[:, :, 0:1, :].reshape(B * self.num_vars, 1, F)
        seq_32 = x_dict[32].reshape(B * self.num_vars, self.p_counts[32], F)
        seq_16 = x_dict[16].reshape(B * self.num_vars, self.p_counts[16], F)
        seq_8  = x_dict[8].reshape(B * self.num_vars, self.p_counts[8], F)
        
        a32 = self.shared_attn if self.shared_layer else self.attn_32
        a16 = self.shared_attn if self.shared_layer else self.attn_16
        a8  = self.shared_attn if self.shared_layer else self.attn_8
        
        q1, _ = a32(query=seq_q, key=seq_32, value=seq_32)
        q1 = seq_q + q1 
        q2, _ = a16(query=q1, key=seq_16, value=seq_16)
        q2 = q1 + q2 
        q3, _ = a8(query=q2, key=seq_8, value=seq_8)
        fused_seq = q2 + q3 
        
        final_repr = fused_seq.squeeze(1).reshape(B, -1)
        return self.linear(self.dropout(final_repr))

class ParallelCrossScaleClassifier(nn.Module):
    def __init__(self, num_vars, num_classes, patches_counts, in_features=384, num_heads=8, shared_layer=True):
        super().__init__()
        self.num_vars = num_vars
        self.p_counts = patches_counts
        self.shared_layer = shared_layer
        
        if shared_layer:
            self.shared_attn = nn.MultiheadAttention(in_features, num_heads, dropout=0.1, batch_first=True)
        else:
            self.attn_32 = nn.MultiheadAttention(in_features, num_heads, dropout=0.1, batch_first=True)
            self.attn_16 = nn.MultiheadAttention(in_features, num_heads, dropout=0.1, batch_first=True)
            self.attn_8  = nn.MultiheadAttention(in_features, num_heads, dropout=0.1, batch_first=True)
        
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(num_vars * in_features, num_classes)

    def forward(self, x):
        B, _, F = x.shape
        start = 0
        x_dict = {}
        for scale in [64, 32, 16, 8]:
            length = self.num_vars * self.p_counts[scale]
            x_dict[scale] = x[:, start : start + length, :]
            start += length
            
        x_64_r = x_dict[64].reshape(B, self.num_vars, self.p_counts[64], F)
        seq_q = x_64_r[:, :, 0:1, :].reshape(B * self.num_vars, 1, F)
        
        x_32_r = x_dict[32].reshape(B, self.num_vars, self.p_counts[32], F)
        x_16_r = x_dict[16].reshape(B, self.num_vars, self.p_counts[16], F)
        x_8_r  = x_dict[8].reshape(B, self.num_vars, self.p_counts[8], F)
        
        if self.shared_layer:
            kv_combined = torch.cat([x_32_r, x_16_r, x_8_r], dim=2) 
            seq_kv = kv_combined.reshape(B * self.num_vars, -1, F)
            fused_seq, _ = self.shared_attn(query=seq_q, key=seq_kv, value=seq_kv)
            fused_seq = seq_q + fused_seq
        else:
            seq_32 = x_32_r.reshape(B * self.num_vars, self.p_counts[32], F)
            seq_16 = x_16_r.reshape(B * self.num_vars, self.p_counts[16], F)
            seq_8  = x_8_r.reshape(B * self.num_vars, self.p_counts[8], F)
            
            out_32, _ = self.attn_32(query=seq_q, key=seq_32, value=seq_32)
            out_16, _ = self.attn_16(query=seq_q, key=seq_16, value=seq_16)
            out_8, _  = self.attn_8(query=seq_q, key=seq_8, value=seq_8)
            fused_seq = seq_q + out_32 + out_16 + out_8
            
        final_repr = fused_seq.squeeze(1).reshape(B, -1)
        return self.linear(self.dropout(final_repr))