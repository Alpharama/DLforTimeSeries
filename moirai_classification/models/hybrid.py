import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from uni2ts.model.moirai import MoiraiModule
from ..encoder import MoiraiEncoder


class DualHybridMeanPoolWrapper(nn.Module):
    """
    Dual-encoder architecture combining fine and coarse temporal representations.

    Mean-pooled features from both encoders are concatenated and used for
    classification.
    """

    def __init__(
        self, num_classes, num_vars=6, size="small", lora_r=8, in_features=384
    ):
        super().__init__()
        self.num_vars = num_vars
        self.in_features = in_features

        # Fine encoder: patch_size = 8, fully fine-tuned
        self.encoder_fine = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=8,
            context_length=36,
            patch_size=8,
            num_samples=100,
            target_dim=num_vars,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        # Coarse encoder: patch_size = 64, LoRA
        enc_coarse = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=64,
            context_length=36,
            patch_size=64,
            num_samples=100,
            target_dim=num_vars,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self.encoder_coarse = get_peft_model(
            enc_coarse,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_r * 2,
                target_modules="all-linear",
            ),
        )

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(num_vars * 2 * in_features, num_classes)

    def forward(self, target, obs, pad):
        B = target.shape[0]
        F = self.in_features

        Z_fine = self.encoder_fine(target, obs, pad)
        P_fine = Z_fine.shape[1] // self.num_vars
        Z_fine = Z_fine.view(B, self.num_vars, P_fine, F)
        fine_pool = Z_fine.mean(dim=2)  # (B, num_vars, F)

        Z_coarse = self.encoder_coarse(target, obs, pad)
        P_coarse = Z_coarse.shape[1] // self.num_vars
        Z_coarse = Z_coarse.view(B, self.num_vars, P_coarse, F)
        coarse_pool = Z_coarse.mean(dim=2)  # (B, num_vars, F)

        combined = torch.cat([fine_pool, coarse_pool], dim=2)  # (B, num_vars, 2*F)
        out = combined.view(B, -1)
        return self.classifier(self.dropout(out))


class DualHybridCoarseToFineWrapper(nn.Module):
    """
    Hybrid dual-encoder model using coarse features to query fine features.

    Cross-attention enriches coarse representations with fine-grained temporal
    information before final classification.
    """

    def __init__(
        self,
        num_classes,
        num_vars=6,
        size="small",
        lora_r=8,
        num_heads=4,
        in_features=384,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.in_features = in_features

        # Coarse encoder: patch_size = 64 with LoRA
        enc_coarse = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=64,
            context_length=36,
            patch_size=64,
            num_samples=100,
            target_dim=num_vars,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self.encoder_coarse = get_peft_model(
            enc_coarse,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_r * 2,
                target_modules="all-linear",
            ),
        )

        # Fine encoder: patch_size = 8, fully fine-tuned
        self.encoder_fine = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=8,
            context_length=36,
            patch_size=8,
            num_samples=100,
            target_dim=num_vars,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(num_vars * in_features, num_classes)

    def forward(self, target, obs, pad):
        B = target.shape[0]
        F = self.in_features

        Z_coarse = self.encoder_coarse(target, obs, pad)
        P_coarse = Z_coarse.shape[1] // self.num_vars
        Z_coarse = Z_coarse.view(B, self.num_vars, P_coarse, F)
        query = Z_coarse[:, :, 0, :]  # (B, num_vars, F)

        Z_fine = self.encoder_fine(target, obs, pad)
        P_fine = Z_fine.shape[1] // self.num_vars
        Z_fine = Z_fine.view(B, self.num_vars, P_fine, F)

        q = query.view(B * self.num_vars, 1, F)
        kv = Z_fine.view(B * self.num_vars, P_fine, F)
        enriched, _ = self.cross_attn(query=q, key=kv, value=kv)
        enriched = enriched.squeeze(1).view(B, self.num_vars, F)
        enriched = enriched + query  # residual

        out = enriched.view(B, -1)
        return self.classifier(self.dropout(out))
