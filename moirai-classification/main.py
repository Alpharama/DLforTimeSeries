# %%
from uni2ts.eval_util.data import get_gluonts_test_dataset
from uni2ts.eval_util.plot import plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# %%
module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

# %%
module
# %%

'''
Module forward
def forward(
    self,
    target: Float[torch.Tensor, "*batch seq_len max_patch"],
    observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
    sample_id: Int[torch.Tensor, "*batch seq_len"],
    time_id: Int[torch.Tensor, "*batch seq_len"],
    variate_id: Int[torch.Tensor, "*batch seq_len"],
    prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
    patch_size: Int[torch.Tensor, "*batch seq_len"],
) -> Distribution:
    """
    Defines the forward pass of MoiraiModule.
    This method expects processed inputs.

    1. Apply scaling to observations
    2. Project from observations to representations
    3. Replace prediction window with learnable mask
    4. Apply transformer layers
    5. Project from representations to distribution parameters
    6. Return distribution object

    :param target: input data
    :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
    :param sample_id: indices indicating the sample index (for packing)
    :param time_id: indices indicating the time index
    :param variate_id: indices indicating the variate index
    :param prediction_mask: binary mask for prediction horizon, 1 if part of the horizon, 0 otherwise
    :param patch_size: patch size for each token
    :return: predictive distribution
    """
    loc, scale = self.scaler(
        target,
        observed_mask * ~prediction_mask.unsqueeze(-1),
        sample_id,
        variate_id,
    ) # Scale the observation
    scaled_target = (target - loc) / scale
    reprs = self.in_proj(scaled_target, patch_size) # Project
    masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight) # Mask 
    reprs = self.encoder(
        masked_reprs,
        packed_attention_mask(sample_id),
        time_id=time_id,
        var_id=variate_id,
    )

    return reprs
'''
