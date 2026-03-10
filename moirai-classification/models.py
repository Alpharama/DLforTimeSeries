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

class MultiScaleFullWrapper(nn.Module):
    """
    Wrapper pour le Full Fine-Tuning Multi-Échelles.
    Instancie correctement 4 encodeurs Moirai (64, 32, 16, 8) et concatène leurs sorties.
    """
    def __init__(self, head_class, head_kwargs, num_vars=6, size="small", remove_last_patch=False):
        super().__init__()
        self.patch_sizes = [64, 32, 16, 8] 
        self.num_vars = num_vars
        self.remove_last_patch = remove_last_patch
        
        # Création correcte des 4 encodeurs Moirai indépendants
        encoders_dict = {}
        for p in self.patch_sizes:
            # Il faut instancier le module HuggingFace pour chaque échelle
            moirai_module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}")
            
            enc = MoiraiEncoder(
                module=moirai_module,
                prediction_length=p,  # On suppose que DO_MASK=True, prediction_length = patch_size
                context_length=36, 
                patch_size=p, 
                num_samples=100, 
                target_dim=num_vars, 
                feat_dynamic_real_dim=0, 
                past_feat_dynamic_real_dim=0
            )
            encoders_dict[str(p)] = enc
            
        self.encoders = nn.ModuleDict(encoders_dict)
        
        # Initialisation de la tête (Flatten, Sequential ou Parallel)
        self.head = head_class(**head_kwargs)

    def forward(self, target, obs, pad):
        features_list = []
        
        for p in self.patch_sizes:
            # 1. Extraction des embeddings pour l'échelle courante
            Z = self.encoders[str(p)](target, obs, pad)
            
            # 2. Retrait du masque (dernier patch) si demandé
            if self.remove_last_patch:
                B, S, F = Z.shape
                P = S // self.num_vars
                # Séparation pour enlever proprement le dernier patch de CHAQUE variable
                Z_reshaped = Z.view(B, self.num_vars, P, F)
                Z_no_mask = Z_reshaped[:, :, :-1, :]
                Z = Z_no_mask.reshape(B, -1, F) # Retour à la shape (Batch, Seq, Features)
                
            features_list.append(Z)
        
        # 3. Concaténation sur la dimension de la séquence (dim=1)
        # Forme résultante : (Batch, Seq_64 + Seq_32 + Seq_16 + Seq_8, Features)
        concat_features = torch.cat(features_list, dim=1)
        
        # 4. Passage dans la tête multi-échelles
        return self.head(concat_features)

class MultiScaleSharedWrapper(nn.Module):
    """
    Wrapper pour le Full Fine-Tuning Multi-Échelles avec UN SEUL encodeur partagé.
    Permet un gain massif de VRAM et force le modèle à apprendre une représentation universelle.
    """
    def __init__(self, head_class, head_kwargs, num_vars=6, size="small", remove_last_patch=False):
        super().__init__()
        self.patch_sizes = [64, 32, 16, 8] 
        self.num_vars = num_vars
        self.remove_last_patch = remove_last_patch
        
        # 1. On charge le Foundation Model UNE SEULE FOIS (Partage des poids)
        shared_moirai = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}")
        
        # 2. On instancie 4 encodeurs qui pointent vers ce MÊME module
        encoders_dict = {}
        for p in self.patch_sizes:
            enc = MoiraiEncoder(
                module=shared_moirai, # <--- LA MAGIE EST ICI
                prediction_length=p,  
                context_length=36, 
                patch_size=p, 
                num_samples=100, 
                target_dim=num_vars, 
                feat_dynamic_real_dim=0, 
                past_feat_dynamic_real_dim=0
            )
            encoders_dict[str(p)] = enc
            
        self.encoders = nn.ModuleDict(encoders_dict)
        
        # Initialisation de la tête
        self.head = head_class(**head_kwargs)

    def forward(self, target, obs, pad):
        features_list = []
        for p in self.patch_sizes:
            # Extraction des embeddings pour l'échelle courante via l'encodeur partagé
            Z = self.encoders[str(p)](target, obs, pad)
            
            # Retrait du masque si demandé
            if self.remove_last_patch:
                B, S, F = Z.shape
                P = S // self.num_vars
                Z_reshaped = Z.view(B, self.num_vars, P, F)
                Z_no_mask = Z_reshaped[:, :, :-1, :]
                Z = Z_no_mask.reshape(B, -1, F)
                
            features_list.append(Z)
        
        # Concaténation des 4 échelles
        concat_features = torch.cat(features_list, dim=1)
        
        return self.head(concat_features)

        
class FlattenMultiScaleHead(nn.Module):
    """
    Tête Baseline : Fait un Mean Pooling sur les patchs de chaque échelle,
    puis concatène (Flatten) le tout avant une couche linéaire.
    """
    def __init__(self, num_vars, num_classes, patches_counts, in_features=384):
        super().__init__()
        self.num_vars = num_vars
        self.p_counts = patches_counts
        
        # Pour chaque échelle, on obtient (num_vars * in_features) après pooling
        # Avec 4 échelles, l'entrée linéaire est 4 * num_vars * in_features
        self.linear = nn.Linear(4 * num_vars * in_features, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B = x.shape[0]
        F = x.shape[-1]
        
        start = 0
        pooled_list = []
        
        # Découpage du tenseur géant et pooling pour chaque échelle
        for scale in [64, 32, 16, 8]:
            length = self.num_vars * self.p_counts[scale]
            x_scale = x[:, start : start + length, :]
            start += length
            
            P = self.p_counts[scale]
            x_reshaped = x_scale.view(B, self.num_vars, P, F)
            
            # Mean Pooling sur la dimension des patchs (dim=2)
            pooled = x_reshaped.mean(dim=2).view(B, -1) 
            pooled_list.append(pooled)
            
        # Concaténation des 4 représentations réduites
        concat_pooled = torch.cat(pooled_list, dim=1)
        return self.linear(self.dropout(concat_pooled))

# Assure-toi d'importer ton MoiraiEncoder correctement
# from encoder import MoiraiEncoder 

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
        use_adalora=False
    ):
        super().__init__()
        
        # 1. Initialisation du modèle de base (Foundation Model)
        self.encoder = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=patch_size, context_length=36, patch_size=patch_size, 
            num_samples=100, target_dim=num_vars, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
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
                total_step=3000
            )
        else:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_r * 2,
                target_modules=target_modules,
                use_dora=use_dora,              # True pour DoRA, False pour LoRA classique
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
            prediction_length=patch_size, context_length=36, patch_size=patch_size, 
            num_samples=100, target_dim=num_vars, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        # No freezing 
        self.model = MoiraiMaskTuner(encoder=moirai_enc, num_vars=num_vars, num_classes=num_classes)

    def forward(self, t, o, p):
        return self.model(t, o, p)

class FullHeadWrapper(nn.Module):
    def __init__(self, head_class, head_kwargs, patch_size, num_vars, size="small", remove_last_patch=False):
        super().__init__()
        moirai_enc = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=patch_size, context_length=36, patch_size=patch_size, 
            num_samples=100, target_dim=num_vars, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        # No freezing!
        head = head_class(**head_kwargs)
        self.model = MoiraiClassifier(encoder=moirai_enc, head=head, remove_last_patch=remove_last_patch, num_vars=num_vars)

    def forward(self, t, o, p):
        return self.model(t, o, p)




class MaskOnlyFinetunerWrapper(nn.Module):
    def __init__(self, patch_size, num_vars, num_classes, size="small"):
        super().__init__()
        # 1. Instanciation de Moirai
        moirai_enc = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=patch_size, context_length=36, patch_size=patch_size, 
            num_samples=100, target_dim=num_vars, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        # 2. Gel de l'encodeur
        unfreeze_only_moirai_mask(moirai_enc)
        # 3. Encapsulation dans ton MaskTuner
        self.model = MoiraiMaskTuner(encoder=moirai_enc, num_vars=num_vars, num_classes=num_classes)

    def forward(self, t, o, p):
        return self.model(t, o, p)

class HeadFinetunerWrapper(nn.Module):
    def __init__(self, head_class, head_kwargs, patch_size, num_vars, size="small", remove_last_patch=False):
        super().__init__()
        # 1. Instanciation de Moirai
        moirai_enc = MoiraiEncoder(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=patch_size, context_length=36, patch_size=patch_size, 
            num_samples=100, target_dim=num_vars, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        # 2. Gel de l'encodeur
        unfreeze_only_moirai_mask(moirai_enc)
        # 3. Création de la tête choisie et Encapsulation
        head = head_class(**head_kwargs)
        self.model = MoiraiClassifier(encoder=moirai_enc, head=head, remove_last_patch=remove_last_patch, num_vars=num_vars)

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
            past_is_pad=past_is_pad
        )
        
        # 2. Gestion du masque de prévision (si on ne veut pas l'inclure dans l'attention)
        if self.remove_last_patch:
            B, S, F = Z.shape
            P = S // self.num_vars
            
            # Séparation pour enlever proprement le dernier patch de CHAQUE variable
            Z_reshaped = Z.view(B, self.num_vars, P, F)
            Z_no_mask = Z_reshaped[:, :, :-1, :]
            Z = Z_no_mask.reshape(B, -1, F) # Retour à la shape (Batch, Sequence, Features)
            
        # 3. La tête prend le relais pour la classification
        logits = self.head(Z)
        
        return logits


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
        # 1. Extraction des embeddings
        Z = self.encoder(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad
        )
        
        B, S, F = Z.shape
        P = S // self.num_vars
        
        # 2. On isole les variables pour ne cibler QUE le dernier patch (le masque)
        Z_reshaped = Z.view(B, self.num_vars, P, F) # .view() is fine here because Z is contiguous
        mask_embeddings = Z_reshaped[:, :, -1, :]   # Slicing makes it non-contiguous!
        
        # 3. Aplatissement et classification directe
        # 💡 SOLUTION: Use .reshape() instead of .view()
        final_repr = mask_embeddings.reshape(B, -1) 
        
        return self.classifier(self.dropout(final_repr))