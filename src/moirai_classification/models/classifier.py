import torch.nn as nn
from uni2ts.model.moirai import MoiraiModule
from ..encoder import MoiraiEncoder
from .utils import unfreeze_only_moirai_mask


class FullHeadWrapper(nn.Module):
    """
    Full fine-tuning model that connects a Moirai encoder to a custom head.

    The entire encoder and head are trained together end-to-end for
    classification or regression tasks.
    """

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

        # No freezing
        head = head_class(**head_kwargs)
        self.model = MoiraiClassifier(
            encoder=moirai_enc,
            head=head,
            remove_last_patch=remove_last_patch,
            num_vars=num_vars,
        )

    def forward(self, t, o, p):
        return self.model(t, o, p)


class HeadFinetunerWrapper(nn.Module):
    """
    Fine-tuning model that freezes most encoder parameters while training a task head.

    Only selected parameters of the encoder remain trainable while the head
    learns to map encoder features to predictions.
    """

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
        unfreeze_only_moirai_mask(moirai_enc)

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
    Generic classifier combining a Moirai encoder with a downstream head module.

    The encoder converts time series inputs into embeddings which are then
    processed by the head to produce classification or regression outputs.
    """

    def __init__(self, encoder, head, remove_last_patch=False, num_vars=6):
        """
        encoder : MoiraiEncoder
        head : Head from heads.py (ex: SingleScaleAttentionClassifier)
        remove_last_patch : Si True, reproduit le comportement 'KEEP_MASK_EMBEDDING=False'
                            en retirant le patch de prévision avant de passer à la tête.
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.remove_last_patch = remove_last_patch
        self.num_vars = num_vars

    def forward(self, past_target, past_observed_target, past_is_pad):

        Z = self.encoder(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )

        if self.remove_last_patch:
            B, S, F = Z.shape
            P = S // self.num_vars

            # Drop Last patch
            Z_reshaped = Z.view(B, self.num_vars, P, F)
            Z_no_mask = Z_reshaped[:, :, :-1, :]
            Z = Z_no_mask.reshape(B, -1, F)  # Shape (Batch, Sequence, Features)

        logits = self.head(Z)

        return logits
