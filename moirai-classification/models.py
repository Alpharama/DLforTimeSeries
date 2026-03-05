import torch
import torch.nn as nn


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