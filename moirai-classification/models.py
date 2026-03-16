import torch
from peft import get_peft_model, LoraConfig, AdaLoraConfig
import torch.nn as nn
from uni2ts.model.moirai import MoiraiModule
from encoder import MoiraiEncoder


def unfreeze_only_moirai_mask(encoder):
    for param in encoder.parameters():
        param.requires_grad = False
    for name, param in encoder.named_parameters():
        if "mask" in name.lower():
            param.requires_grad = True


class LoraHeadWrapper(nn.Module):
    def __init__(
        self,
        head_class,
        head_kwargs,
        patch_size,
        num_vars,
        size,
        remove_last_patch=False,
        lora_r=8,
        use_dora=False,
        use_adalora=False,
    ):
        super().__init__()

        # 1. Initialisation du modèle de base (Foundation Model)
        self.encoder = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=patch_size,
            context_length=36,
            patch_size=patch_size,
            num_samples=100,
            target_dim=num_vars,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        # Cibles classiques pour les couches d'attention (à adapter si Moirai a des noms différents)
        # Par exemple : ["q_proj", "v_proj", "k_proj", "out_proj"] ou ["q", "v"]
        target_modules = "all-linear"

        if use_adalora:
            peft_config = AdaLoraConfig(
                target_r=lora_r,
                init_r=lora_r + 4,
                lora_alpha=lora_r * 2,
                target_modules=target_modules,
                tinit=200,
                tfinal=1000,
                deltaT=10,
                total_step=3000,
            )
        else:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_r * 2,
                target_modules=target_modules,
                use_dora=use_dora,  # True pour DoRA, False pour LoRA classique
                # lora_dropout=0.05
            )

        # Envelopper l'encodeur avec la configuration PEFT
        self.encoder = get_peft_model(self.encoder, peft_config)

        # 3. Initialisation de la tête de classification/régression
        self.head = head_class(**head_kwargs)

    def forward(self, target, obs, pad):
        # Passage des 3 arguments dans l'encodeur (qui est maintenant un modèle PEFT)
        features = self.encoder(target, obs, pad)
        # Passage dans la tête spécifique
        out = self.head(features)
        return out


class FullMaskOnlyWrapper(nn.Module):
    def __init__(self, patch_size, num_vars, num_classes, size="small"):
        super().__init__()
        moirai_enc = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=patch_size,
            context_length=36,
            patch_size=patch_size,
            num_samples=100,
            target_dim=num_vars,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        # No freezing
        self.model = MoiraiMaskTuner(
            encoder=moirai_enc, num_vars=num_vars, num_classes=num_classes
        )

    def forward(self, t, o, p):
        return self.model(t, o, p)


class FullHeadWrapper(nn.Module):
    def __init__(
        self,
        head_class,
        head_kwargs,
        patch_size,
        num_vars,
        size="small",
        remove_last_patch=False,
    ):
        super().__init__()
        moirai_enc = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=patch_size,
            context_length=36,
            patch_size=patch_size,
            num_samples=100,
            target_dim=num_vars,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        # No freezing!
        head = head_class(**head_kwargs)
        self.model = MoiraiClassifier(
            encoder=moirai_enc,
            head=head,
            remove_last_patch=remove_last_patch,
            num_vars=num_vars,
        )

    def forward(self, t, o, p):
        return self.model(t, o, p)


class MaskOnlyFinetunerWrapper(nn.Module):
    def __init__(self, patch_size, num_vars, num_classes, size="small"):
        super().__init__()
        # 1. Instanciation de Moirai
        moirai_enc = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=patch_size,
            context_length=36,
            patch_size=patch_size,
            num_samples=100,
            target_dim=num_vars,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        # 2. Gel de l'encodeur
        unfreeze_only_moirai_mask(moirai_enc)
        # 3. Encapsulation dans ton MaskTuner
        self.model = MoiraiMaskTuner(
            encoder=moirai_enc, num_vars=num_vars, num_classes=num_classes
        )

    def forward(self, t, o, p):
        return self.model(t, o, p)


class HeadFinetunerWrapper(nn.Module):
    def __init__(
        self,
        head_class,
        head_kwargs,
        patch_size,
        num_vars,
        size="small",
        remove_last_patch=False,
    ):
        super().__init__()
        # 1. Instanciation de Moirai
        moirai_enc = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=patch_size,
            context_length=36,
            patch_size=patch_size,
            num_samples=100,
            target_dim=num_vars,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        # 2. Gel de l'encodeur
        unfreeze_only_moirai_mask(moirai_enc)
        # 3. Création de la tête choisie et Encapsulation
        head = head_class(**head_kwargs)
        self.model = MoiraiClassifier(
            encoder=moirai_enc,
            head=head,
            remove_last_patch=remove_last_patch,
            num_vars=num_vars,
        )

    def forward(self, t, o, p):
        return self.model(t, o, p)


class MoiraiClassifier(nn.Module):
    """
    Modèle End-to-End classique : Combine l'encodeur Moirai et une tête de heads.py
    """

    def __init__(self, encoder, head, remove_last_patch=False, num_vars=6):
        """
        encoder : Instance de MoiraiEncoder
        head : Instance d'une tête issue de heads.py (ex: SingleScaleAttentionClassifier)
        remove_last_patch : Si True, reproduit le comportement 'KEEP_MASK_EMBEDDING=False'
                            en retirant le patch de prévision avant de passer à la tête.
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.remove_last_patch = remove_last_patch
        self.num_vars = num_vars

    def forward(self, past_target, past_observed_target, past_is_pad):
        # 1. L'encodeur transforme les séries temporelles brutes en embeddings (Z)
        Z = self.encoder(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )

        # 2. Gestion du masque de prévision (si on ne veut pas l'inclure dans l'attention)
        if self.remove_last_patch:
            B, S, F = Z.shape
            P = S // self.num_vars

            # Séparation pour enlever proprement le dernier patch de CHAQUE variable
            Z_reshaped = Z.view(B, self.num_vars, P, F)
            Z_no_mask = Z_reshaped[:, :, :-1, :]
            Z = Z_no_mask.reshape(
                B, -1, F
            )  # Retour à la shape (Batch, Sequence, Features)

        # 3. La tête prend le relais pour la classification
        logits = self.head(Z)

        return logits


class DualHybridMeanPoolWrapper(nn.Module):
    """
    Dual encoder mean pooling: FullFT p8 + LoRA p64.

    Architecture:
      1. Fine encoder (patch 8, fully fine-tuned):
         Produces Z_fine of shape (B, num_vars * P_fine, F).
         Mean pooling over P_fine patches per variable → (B, num_vars, F).

      2. Coarse encoder (patch 64, LoRA r=lora_r):
         Produces Z_coarse of shape (B, num_vars * P_coarse, F).
         Mean pooling over P_coarse patches per variable → (B, num_vars, F).

      3. Concatenate fine and coarse mean-pooled representations per variable:
         → (B, num_vars, 2*F), flattened to (B, num_vars * 2 * F).

      4. Dropout → Linear → num_classes.
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
    Hybrid dual encoder cascade: LoRA p64 (coarse) + FullFT p8 (fine).

    Architecture:
      1. Coarse encoder (patch 64, LoRA): first patch per variable → query.
      2. Fine encoder (patch 8, fully fine-tuned): all patches → key/value.
      3. Cross-attention (coarse query → fine key/value) + residual.
      4. Dropout → Linear → num_classes.
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


class MoiraiMaskTuner(nn.Module):
    """
    Modèle spécifique au Mask Tuning :
    On ne classifie qu'en utilisant le patch de prévision (le masque).
    """

    def __init__(self, encoder, num_vars, num_classes, in_features=384):
        super().__init__()
        self.encoder = encoder
        self.num_vars = num_vars

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(num_vars * in_features, num_classes)

    def forward(self, past_target, past_observed_target, past_is_pad):
        Z = self.encoder(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )

        B, S, F = Z.shape
        P = S // self.num_vars

        Z_reshaped = Z.view(B, self.num_vars, P, F)
        mask_embeddings = Z_reshaped[:, :, -1, :]

        final_repr = mask_embeddings.reshape(B, -1)

        return self.classifier(self.dropout(final_repr))
