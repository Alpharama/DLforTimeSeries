import torch
import copy


def train_finetune(
    model,
    train_loader,
    val_loader,
    lr,
    epochs=500,
    weight_decay=0.01,
    device="cuda",
    verbose=False,
):
    """
    Train a model using supervised fine-tuning with early stopping based on validation loss.
    """
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    patience = 5
    epochs_no_improve = 0
    best_avg_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())

    global_step = 0

    for epoch in range(epochs):
        model.train()
        for b_target, b_obs, b_pad, b_y in train_loader:
            optimizer.zero_grad()
            logits = model(b_target, b_obs, b_pad)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            global_step += 1
            # On vérifie si le modèle possède la structure d'AdaLoRA avant d'appeler la fonction
            if (
                hasattr(model, "encoder")
                and hasattr(model.encoder, "base_model")
                and hasattr(model.encoder.base_model, "update_and_allocate")
            ):
                model.encoder.base_model.update_and_allocate(global_step)

        model.eval()
        total_val_loss, total = 0.0, 0
        with torch.no_grad():
            for b_target, b_obs, b_pad, b_y in val_loader:
                logits = model(b_target, b_obs, b_pad)
                loss = criterion(logits, b_y)
                total_val_loss += loss.item() * b_y.size(0)
                total += b_y.size(0)

        avg_val_loss = total_val_loss / total

        if verbose:
            print(avg_val_loss)

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
