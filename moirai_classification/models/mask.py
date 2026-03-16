import torch.nn as nn
from uni2ts.model.moirai import MoiraiModule
from ..encoder import MoiraiEncoder
from .utils import unfreeze_only_moirai_mask


class FullMaskOnlyWrapper(nn.Module):
    """
    Mask-based classification model using a fully trainable Moirai encoder.

    Predictions are computed using only the mask patch embeddings produced
    by the encoder.
    """

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


class MaskOnlyFinetunerWrapper(nn.Module):
    """
    Mask-only fine-tuning model where most encoder parameters are frozen.

    Only mask-related parameters are updated while the classifier learns
    from the mask embeddings.
    """

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

        unfreeze_only_moirai_mask(moirai_enc)

        self.model = MoiraiMaskTuner(
            encoder=moirai_enc, num_vars=num_vars, num_classes=num_classes
        )

    def forward(self, t, o, p):
        return self.model(t, o, p)


class MoiraiMaskTuner(nn.Module):
    """
    Model that performs classification using the mask embeddings from Moirai.

    The mask tokens corresponding to each variable are extracted and passed
    through a linear classifier to produce predictions.
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
