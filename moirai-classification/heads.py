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