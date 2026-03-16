import torch
import copy
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .train import train_finetune


def grid_search_heads(
    head_class,
    head_kwargs,
    train_loader_z,
    val_loader_z,
    test_loader_z,
    lr_grid=[1e-3, 1e-4],
    wd_grid=[0.01, 0.05],
    epochs=500,
    device="cuda",
):
    """
    Perform grid search over learning rate and weight decay for training a prediction head.
    """
    best_overall_val_loss = float("inf")
    best_model = None
    criterion = nn.CrossEntropyLoss()

    for wd in wd_grid:
        for lr in lr_grid:
            head = head_class(**head_kwargs).to(device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

            best_val_loss_local = float("inf")
            best_head_weights = None
            epochs_no_improve = 0
            patience = 20

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

                if epochs_no_improve >= patience:
                    break

            if best_val_loss_local < best_overall_val_loss:
                best_overall_val_loss = best_val_loss_local
                head.load_state_dict(best_head_weights)
                best_model = copy.deepcopy(head)

    best_model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for b_z, b_y in test_loader_z:
            b_z, b_y = b_z.to(device), b_y.to(device)
            preds = torch.argmax(best_model(b_z), dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(b_y.cpu().numpy())

    metrics = {"Accuracy": accuracy_score(all_targets, all_preds)}

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    metrics.update(
        {
            "Macro Precision": prec_macro,
            "Macro Recall": rec_macro,
            "Macro F1": f1_macro,
            "Weighted Precision": prec_weight,
            "Weighted Recall": rec_weight,
            "Weighted F1": f1_weight,
        }
    )

    return best_model, metrics


def universal_grid_search(
    model_class,
    model_kwargs,
    train_loader,
    val_loader,
    test_loader,
    lr_grid=[1e-4, 5e-5],
    wd_grid=[0.01, 0.05],
    epochs=500,
    device="cuda",
    verbose=False,
    patch_size=None,
):
    """
    Perform grid search over training hyperparameters to select the best full model configuration.
    """
    best_overall_val_loss = float("inf")
    best_model = None

    for wd in wd_grid:
        for lr in lr_grid:
            print(f"LR={lr}| WD={wd}")

            model = model_class(**model_kwargs).to(device)

            val_loss, trained_model = train_finetune(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=lr,
                epochs=epochs,
                weight_decay=wd,
                device=device,
                verbose=verbose,
            )

            if val_loss < best_overall_val_loss:
                best_overall_val_loss = val_loss
                best_model = copy.deepcopy(trained_model)
            print(f"Val Loss: {val_loss:.4f}")

    best_model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for b_t, b_o, b_p, b_y in test_loader:
            b_t, b_o, b_p = b_t.to(device), b_o.to(device), b_p.to(device)

            logits = best_model(b_t, b_o, b_p)
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(b_y.cpu().numpy())

    metrics = {"Accuracy": accuracy_score(all_targets, all_preds)}

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    metrics.update(
        {
            "Macro Precision": prec_macro,
            "Macro Recall": rec_macro,
            "Macro F1": f1_macro,
            "Weighted Precision": prec_weight,
            "Weighted Recall": rec_weight,
            "Weighted F1": f1_weight,
        }
    )

    return best_model, metrics
