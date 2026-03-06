import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import LabelEncoder




NUM_VARS = 6

#@torch.no_grad()
def get_z_loaders(encoder, tr_loader, va_loader, te_loader, head_batch_size=256, device="cuda", remove_last_patch=True, num_vars=6):
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
    
    tr_z_loader = DataLoader(TensorDataset(Z_tr, y_tr), batch_size=head_batch_size, shuffle=True)
    va_z_loader = DataLoader(TensorDataset(Z_va, y_va), batch_size=head_batch_size, shuffle=False)
    te_z_loader = DataLoader(TensorDataset(Z_te, y_te), batch_size=head_batch_size, shuffle=False)
    
    return tr_z_loader, va_z_loader, te_z_loader


def grid_search_heads(
    head_class, head_kwargs, train_loader_z, val_loader_z, test_loader_z, 
    lr_grid=[1e-3, 1e-4], wd_grid=[0.01, 0.05], epochs=50, device="cuda"
):
    best_overall_val_loss = float('inf')
    best_model = None
    criterion = nn.CrossEntropyLoss()
    
    for wd in wd_grid:
        for lr in lr_grid:
            head = head_class(**head_kwargs).to(device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
            
            best_val_loss_local = float('inf')
            best_head_weights = None
            epochs_no_improve = 0
            patience = 15
            
            for epoch in range(epochs):
                head.train()
                for b_z, b_y in train_loader_z:
                    b_z, b_y = b_z.to(device), b_y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(head(b_z), b_y)
                    loss.backward()
                    optimizer.step()
                    
                head.eval()
                total_val_loss, total = 0.0, 0
                with torch.no_grad():
                    for b_z, b_y in val_loader_z:
                        b_z, b_y = b_z.to(device), b_y.to(device)
                        loss = criterion(head(b_z), b_y)
                        total_val_loss += loss.item() * b_y.size(0)
                        total += b_y.size(0)
                
                avg_val_loss = total_val_loss / total
                if avg_val_loss < best_val_loss_local:
                    best_val_loss_local = avg_val_loss
                    best_head_weights = copy.deepcopy(head.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience: break
                    
            if best_val_loss_local < best_overall_val_loss:
                best_overall_val_loss = best_val_loss_local
                head.load_state_dict(best_head_weights)
                best_model = copy.deepcopy(head)
                
    best_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b_z, b_y in test_loader_z:
            b_z, b_y = b_z.to(device), b_y.to(device)
            preds = torch.argmax(best_model(b_z), dim=-1)
            correct += (preds == b_y).sum().item()
            total += b_y.size(0)
            
    return best_model, correct / total







def universal_grid_search(
    model_class, 
    model_kwargs, 
    train_loader, 
    val_loader, 
    test_loader, 
    lr_grid=[1e-4, 5e-5], 
    wd_grid=[0.01, 0.05], 
    epochs=50,
    device="cuda"
):
    best_overall_val_loss = float('inf')
    best_model = None
    
    
    for wd in wd_grid:
        for lr in lr_grid:
            print(f"LR={lr} | WD={wd}")
            
            model = model_class(**model_kwargs).to(device)
            
            val_loss, trained_model = train_finetune(
                model=model, 
                train_loader=train_loader, 
                val_loader=val_loader,
                lr=lr, 
                epochs=epochs, 
                weight_decay=wd, 
                device=device
            )
            
            if val_loss < best_overall_val_loss:
                best_overall_val_loss = val_loss
                best_model = copy.deepcopy(trained_model)
                print(f"Val Loss: {val_loss:.4f}")
            
    best_model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for b_t, b_o, b_p, b_y in test_loader:
            b_t, b_o, b_p, b_y = b_t.to(device), b_o.to(device), b_p.to(device), b_y.to(device)
            
            logits = best_model(b_t, b_o, b_p)
            predictions = torch.argmax(logits, dim=-1)
            
            correct += (predictions == b_y).sum().item()
            total += b_y.size(0)
            
    test_acc = correct / total
    print(f"Acc on Test Set : {test_acc:.4f}\n")
    
    return best_model, test_acc







def unfreeze_only_moirai_mask(encoder):
    for param in encoder.parameters():
        param.requires_grad = False
    for name, param in encoder.named_parameters():
        if "mask" in name.lower():
            param.requires_grad = True


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

    tr_loader, va_loader = create_raw_dataloaders(X_tr_t, X_tr_o, X_tr_p, y_tr_t, batch_size=batch_size, device=device)
    te_loader = DataLoader(TensorDataset(X_te_t, X_te_o, X_te_p, y_te_t), batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader, te_loader, num_classes




def grid_search_finetune(
    model_class, model_kwargs, train_loader, val_loader, test_loader, device="cuda"
):
    # Grille de paramètres adaptée au Fine-Tuning
    lr_grid = [1e-4, 5e-5]
    wd_grid = [0.01, 0.05]
    
    best_overall_val_loss = float('inf')
    best_model = None
    
    for wd in wd_grid:
        for lr in lr_grid:
            print(f"      [GridSearch] Test avec LR={lr} | WD={wd}")
            
            # 💡 C'est ici que le modèle se ré-instancie à neuf !
            model = model_class(**model_kwargs).to(device)
            
            val_loss, trained_model = train_finetune(
                model=model, train_loader=train_loader, val_loader=val_loader,
                lr=lr, epochs=50, weight_decay=wd, device=device
            )
            
            if val_loss < best_overall_val_loss:
                best_overall_val_loss = val_loss
                best_model = copy.deepcopy(trained_model)     
            
    # Évaluation du meilleur modèle sur le test set
    best_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b_t, b_o, b_p, b_y in test_loader:
            b_t, b_o, b_p, b_y = b_t.to(device), b_o.to(device), b_p.to(device), b_y.to(device)
            logits = best_model(b_t, b_o, b_p)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == b_y).sum().item()
            total += b_y.size(0)
            
    test_acc = correct / total
    return best_model, test_acc




def apply_pooling_pt(Z_tensor, method, num_vars=NUM_VARS):
    N, S, F = Z_tensor.shape
    P = S // num_vars # Calcul automatique du nombre de patches par variable
    
    # On reshape le tenseur pour séparer les Variables et les Patches
    # Forme résultante : (Batch, Variables, Patches, Features)
    Z_reshaped = Z_tensor.view(N, num_vars, P, F)
    
    # Basique et Global
    if method == "flatten":
        return Z_tensor.reshape(N, -1)
        
    elif method == "global_mean":
        return Z_tensor.mean(dim=1)
        
    elif method == "global_max":
        return Z_tensor.max(dim=1).values
        
    elif method == "global_min":
        return Z_tensor.min(dim=1).values
    
    elif method == "global_mean_max_min":
        return torch.cat([
            Z_tensor.mean(dim=1),
            Z_tensor.max(dim=1).values,
            Z_tensor.min(dim=1).values
        ], dim=1)

    # Pooling sur les Patches (on garde les variables distinctes) ---
    # Réduction sur la dimension 2 (Patches). Résultat : (N, num_vars, F), puis on aplatit
    elif method == "mean_over_patches":
        return Z_reshaped.mean(dim=2).reshape(N, -1)
        
    elif method == "max_over_patches":
        return Z_reshaped.max(dim=2).values.reshape(N, -1)
        
    elif method == "min_over_patches":
        return Z_reshaped.min(dim=2).values.reshape(N, -1)
        
    elif method == "mean_max_min_over_patches":
        p_mean = Z_reshaped.mean(dim=2).reshape(N, -1)
        p_max  = Z_reshaped.max(dim=2).values.reshape(N, -1)
        p_min  = Z_reshaped.min(dim=2).values.reshape(N, -1)
        return torch.cat([p_mean, p_max, p_min], dim=1)

    # Pooling sur les Variables (on synchronise les patches entre variables) ---
    # Réduction sur la dimension 1 (Variables). Résultat : (N, P, F), puis on aplatit
    elif method == "mean_over_variables":
        return Z_reshaped.mean(dim=1).reshape(N, -1)
        
    elif method == "max_over_variables":
        return Z_reshaped.max(dim=1).values.reshape(N, -1)
        
    elif method == "min_over_variables":
        return Z_reshaped.min(dim=1).values.reshape(N, -1)
        
    elif method == "mean_max_min_over_variables":
        v_mean = Z_reshaped.mean(dim=1).reshape(N, -1)
        v_max  = Z_reshaped.max(dim=1).values.reshape(N, -1)
        v_min  = Z_reshaped.min(dim=1).values.reshape(N, -1)
        return torch.cat([v_mean, v_max, v_min], dim=1)

    else:
        raise ValueError(f"Method {method} unknow")

pooling_methods = [
    "flatten", 
    "global_mean", "global_max", "global_min", "global_mean_max_min",
    "mean_over_patches", "max_over_patches", "min_over_patches", "mean_max_min_over_patches",
    "mean_over_variables", "max_over_variables", "min_over_variables", "mean_max_min_over_variables"
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

# ==========================================
# 1. FONCTIONS D'ENTRAÎNEMENT
# ==========================================
def train(
    model, train_loader, val_loader, lr, 
    epochs=100, weight_decay=0.005, device="cuda"
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    patience = 50
    epochs_no_improve = 0
    best_avg_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        for batch_z, batch_y in train_loader:
            batch_z, batch_y = batch_z.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_z)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0.0
        total = 0
        
        with torch.no_grad():
            for batch_z, batch_y in val_loader:
                batch_z, batch_y = batch_z.to(device), batch_y.to(device)
                logits = model(batch_z)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item() * batch_y.size(0)
                total += batch_y.size(0)
                
        avg_val_loss = total_val_loss / total
        
        # Early stopping logic
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            # print(f'Early stop at epoch {epoch}') # Décommenter pour debug
            break
            
    model.load_state_dict(best_model_weights)
    return best_avg_val_loss, model

def grid_search_attention_model(
    model_class, model_kwargs, train_loader, val_loader, test_loader, device="cuda", f_train = train
):
    lr_grid = [1e-4]
    wd_grid = [0.0, 0.05, 0.1, 0.2] 
    
    best_overall_val_loss = float('inf')
    best_lr = None
    best_wd = None
    best_model = None
    
    for wd in wd_grid:
        for lr in lr_grid:
            model = model_class(**model_kwargs).to(device)
            val_loss, trained_model = f_train(
                model=model, train_loader=train_loader, val_loader=val_loader,
                lr=lr, epochs=500, weight_decay=wd, device=device
            )
            
            if val_loss < best_overall_val_loss:
                best_overall_val_loss = val_loss
                best_lr = lr
                best_wd = wd
                best_model = copy.deepcopy(trained_model)     
            
    best_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_z, batch_y in test_loader:
            batch_z, batch_y = batch_z.to(device), batch_y.to(device)
            logits = best_model(batch_z)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
            
    test_acc = correct / total
    return best_model, test_acc


# ==========================================
# 2. GÉNÉRATEURS DE DATALOADERS (Sécurisés et Indépendants)
# ==========================================
def create_single_scale_dataloaders(Z_train, Z_test, y_train, y_test, batch_size=256, device="cuda"):
    """Crée les dataloaders pour une seule taille de patch (ex: juste 16)"""
    Z_train_np = Z_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    
    Z_tr_split, Z_va_split, y_tr_split, y_va_split = train_test_split(
        Z_train_np, y_train_np, test_size=0.2, random_state=42, stratify=y_train_np
    )
    
    Z_tr = torch.tensor(Z_tr_split, device=device)
    y_tr = torch.tensor(y_tr_split, dtype=torch.long, device=device)
    Z_va = torch.tensor(Z_va_split, device=device)
    y_va = torch.tensor(y_va_split, dtype=torch.long, device=device)
    
    tr_loader = DataLoader(TensorDataset(Z_tr, y_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Z_va, y_va), batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(TensorDataset(Z_test.to(device), y_test.to(device)), batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader, te_loader

def create_all_scales_dataloaders(Z_train_dict, Z_test_dict, scales, y_train, y_test, batch_size=256, device="cuda"):
    """Crée les dataloaders pour la combinaison de toutes les tailles (ex: 64, 32, 16, 8 concaténés)"""
    # Concaténation de toutes les échelles demandées
    Z_tr_comb = torch.cat([Z_train_dict[s] for s in scales], dim=1).cpu().numpy()
    Z_te_comb = torch.cat([Z_test_dict[s] for s in scales], dim=1).to(device)
    y_train_np = y_train.cpu().numpy()
    
    Z_tr_split, Z_va_split, y_tr_split, y_va_split = train_test_split(
        Z_tr_comb, y_train_np, test_size=0.2, random_state=42, stratify=y_train_np
    )
    
    Z_tr = torch.tensor(Z_tr_split, device=device)
    y_tr = torch.tensor(y_tr_split, dtype=torch.long, device=device)
    Z_va = torch.tensor(Z_va_split, device=device)
    y_va = torch.tensor(y_va_split, dtype=torch.long, device=device)
    
    tr_loader = DataLoader(TensorDataset(Z_tr, y_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Z_va, y_va), batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(TensorDataset(Z_te_comb, y_test.to(device)), batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader, te_loader

def create_raw_dataloaders(
    X_target, X_obs, X_pad, y, 
    batch_size=64, # ⚠️ Attention : batch_size plus petit car Moirai consomme de la VRAM !
    device="cuda"
):
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
    
    tr_loader = DataLoader(TensorDataset(t_tr, o_tr, p_tr, y_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(t_va, o_va, p_va, y_va), batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader

def train_finetune(
    model, train_loader, val_loader, lr, 
    epochs=500, weight_decay=0.01, device="cuda"
):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    patience = 30
    epochs_no_improve = 0
    best_avg_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        for b_target, b_obs, b_pad, b_y in train_loader:
            optimizer.zero_grad()
            logits = model(b_target, b_obs, b_pad)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss, total = 0.0, 0
        with torch.no_grad():
            for b_target, b_obs, b_pad, b_y in val_loader:
                logits = model(b_target, b_obs, b_pad)
                loss = criterion(logits, b_y)
                total_val_loss += loss.item() * b_y.size(0)
                total += b_y.size(0)
                
        avg_val_loss = total_val_loss / total
        
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f" Early stopping : {epoch}")
            break
            
    model.load_state_dict(best_model_weights)
    return best_avg_val_loss, model