import torch
import torch.nn as nn
import numpy as np
import copy
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

NUM_VARS = 6


@torch.no_grad()
def get_z_loaders(
    encoder,
    tr_loader,
    va_loader,
    te_loader,
    head_batch_size=256,
    device="cuda",
    remove_last_patch=True,
    num_vars=6,
):
    encoder.eval()
    encoder.to(device)

    def process_loader(loader):
        Z_list, y_list = [], []
        for b_t, b_o, b_p, b_y in loader:
            b_t, b_o, b_p = b_t.to(device), b_o.to(device), b_p.to(device)
            Z = encoder(b_t, b_o, b_p)

            if remove_last_patch:
                B, S, F = Z.shape
                P = S // num_vars
                Z_reshaped = Z.view(B, num_vars, P, F)
                Z_no_mask = Z_reshaped[:, :, :-1, :]
                Z = Z_no_mask.reshape(B, -1, F)

            Z_list.append(Z.cpu())
            y_list.append(b_y.cpu())
        return torch.cat(Z_list), torch.cat(y_list)

    Z_tr, y_tr = process_loader(tr_loader)
    Z_va, y_va = process_loader(va_loader)
    Z_te, y_te = process_loader(te_loader)

    _g = torch.Generator()
    _g.manual_seed(SEED)
    tr_z_loader = DataLoader(
        TensorDataset(Z_tr, y_tr),
        batch_size=head_batch_size,
        shuffle=True,
        generator=_g,
    )
    va_z_loader = DataLoader(
        TensorDataset(Z_va, y_va), batch_size=head_batch_size, shuffle=False
    )
    te_z_loader = DataLoader(
        TensorDataset(Z_te, y_te), batch_size=head_batch_size, shuffle=False
    )

    return tr_z_loader, va_z_loader, te_z_loader


def get_lsst_dataloaders(batch_size, device="cuda"):
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    num_classes = len(set(y_train_encoded))

    X_tr_t, X_tr_o, X_tr_p = preprocess_data(X_train, device=device)
    X_te_t, X_te_o, X_te_p = preprocess_data(X_test, device=device)

    y_tr_t = torch.tensor(y_train_encoded, dtype=torch.long, device=device)
    y_te_t = torch.tensor(y_test_encoded, dtype=torch.long, device=device)

    tr_loader, va_loader = create_raw_dataloaders(
        X_tr_t, X_tr_o, X_tr_p, y_tr_t, batch_size=batch_size, device=device
    )
    te_loader = DataLoader(
        TensorDataset(X_te_t, X_te_o, X_te_p, y_te_t),
        batch_size=batch_size,
        shuffle=False,
    )

    return tr_loader, va_loader, te_loader, num_classes


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


pooling_methods = [
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


def preprocess_data(
    data: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    data: np.ndarray of shape (N, T, V) = (n_individual, time, variate)
    Assumes NO missing values and NO padding in the raw data.
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    if data.ndim != 3:
        raise ValueError(f"Expected shape (N,T,V), got {data.shape}")

    N, T, V = data.shape

    # (N,T,V)
    past_target = torch.as_tensor(data, dtype=dtype, device=device)

    # observed mask: (N,T,V) all True no missing values
    past_observed_target = torch.ones((N, T, V), dtype=torch.bool, device=device)

    # padding mask: (N,T) all if no padding
    past_is_pad = torch.zeros((N, T), dtype=torch.bool, device=device)

    return past_target, past_observed_target, past_is_pad


def create_raw_dataloaders(X_target, X_obs, X_pad, y, batch_size=64, device="cuda"):
    X_target_np = X_target.cpu().numpy()
    X_obs_np = X_obs.cpu().numpy()
    X_pad_np = X_pad.cpu().numpy()
    y_np = y.cpu().numpy()

    # Stratified Split
    indices = np.arange(len(y_np))
    idx_tr, idx_va, y_tr, y_va = train_test_split(
        indices, y_np, test_size=0.2, random_state=42, stratify=y_np
    )

    # Reconstruction des tenseurs Train
    t_tr = torch.tensor(X_target_np[idx_tr], device=device)
    o_tr = torch.tensor(X_obs_np[idx_tr], device=device)
    p_tr = torch.tensor(X_pad_np[idx_tr], device=device)
    y_tr = torch.tensor(y_tr, dtype=torch.long, device=device)

    # Reconstruction des tenseurs Val
    t_va = torch.tensor(X_target_np[idx_va], device=device)
    o_va = torch.tensor(X_obs_np[idx_va], device=device)
    p_va = torch.tensor(X_pad_np[idx_va], device=device)
    y_va = torch.tensor(y_va, dtype=torch.long, device=device)

    _g = torch.Generator()
    _g.manual_seed(SEED)
    tr_loader = DataLoader(
        TensorDataset(t_tr, o_tr, p_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        generator=_g,
    )
    va_loader = DataLoader(
        TensorDataset(t_va, o_va, p_va, y_va), batch_size=batch_size, shuffle=False
    )

    return tr_loader, va_loader
