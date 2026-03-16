from peft import get_peft_model, LoraConfig, AdaLoraConfig
import torch.nn as nn
from uni2ts.model.moirai import MoiraiModule
from ..encoder import MoiraiEncoder


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
