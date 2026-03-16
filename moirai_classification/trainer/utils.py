import torch
import numpy as np
import random

SEED = 42
NUM_VARS = 6

POOLING_METHODS = [
    "flatten",
    "global_mean",
    "global_max",
    "global_min",
    "global_mean_max_min",
    "mean_over_patches",
    "max_over_patches",
    "min_over_patches",
    "mean_max_min_over_patches",
    "mean_over_variables",
    "max_over_variables",
    "min_over_variables",
    "mean_max_min_over_variables",
]


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for full reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_pooling_pt(Z_tensor, method, num_vars=NUM_VARS):
    N, S, F = Z_tensor.shape
    P = S // num_vars

    Z_reshaped = Z_tensor.view(N, num_vars, P, F)

    if method == "flatten":
        return Z_tensor.reshape(N, -1)

    elif method == "global_mean":
        return Z_tensor.mean(dim=1)

    elif method == "global_max":
        return Z_tensor.max(dim=1).values

    elif method == "global_min":
        return Z_tensor.min(dim=1).values

    elif method == "global_mean_max_min":
        return torch.cat(
            [
                Z_tensor.mean(dim=1),
                Z_tensor.max(dim=1).values,
                Z_tensor.min(dim=1).values,
            ],
            dim=1,
        )

    elif method == "mean_over_patches":
        return Z_reshaped.mean(dim=2).reshape(N, -1)

    elif method == "max_over_patches":
        return Z_reshaped.max(dim=2).values.reshape(N, -1)

    elif method == "min_over_patches":
        return Z_reshaped.min(dim=2).values.reshape(N, -1)

    elif method == "mean_max_min_over_patches":
        p_mean = Z_reshaped.mean(dim=2).reshape(N, -1)
        p_max = Z_reshaped.max(dim=2).values.reshape(N, -1)
        p_min = Z_reshaped.min(dim=2).values.reshape(N, -1)
        return torch.cat([p_mean, p_max, p_min], dim=1)

    elif method == "mean_over_variables":
        return Z_reshaped.mean(dim=1).reshape(N, -1)

    elif method == "max_over_variables":
        return Z_reshaped.max(dim=1).values.reshape(N, -1)

    elif method == "min_over_variables":
        return Z_reshaped.min(dim=1).values.reshape(N, -1)

    elif method == "mean_max_min_over_variables":
        v_mean = Z_reshaped.mean(dim=1).reshape(N, -1)
        v_max = Z_reshaped.max(dim=1).values.reshape(N, -1)
        v_min = Z_reshaped.min(dim=1).values.reshape(N, -1)
        return torch.cat([v_mean, v_max, v_min], dim=1)

    else:
        raise ValueError(f"Method {method} unknow")
