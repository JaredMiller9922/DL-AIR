import torch
from utils.model_utils.losses import pit_mse_loss, mse_loss

def train_model(model, train_loader, val_loader, plotter, epochs=300, device="cpu", lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_hist, val_hist = [] , []

    for epoch in range(1, epochs + 1):
        # --- Snapshot Logic ---
        if epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                # Take one batch for the training snapshot
                sample = next(iter(train_loader))
                x_snap, y_snap = sample["x"].to(device), sample["y"].to(device)
                pred_snap = model(x_snap)
                # Now plotter is in scope!
                plotter.plot_separation_performance(y_snap, pred_snap, 
                    model_name=f"{model.__class__.__name__}_Train_Epoch_{epoch}")
        
        model.train()
        total_train = 0
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            pred = model(x)
            loss = pit_mse_loss(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        
        # Validation Phase
        model.eval()
        total_val = 0
        with torch.no_grad():
            for v_batch in val_loader:
                vx, vy = v_batch["x"].to(device), v_batch["y"].to(device)
                v_pred = model(vx)
                total_val += mse_loss(v_pred, vy).item()
        
        avg_val = total_val / len(val_loader)
        train_hist.append(avg_train)
        val_hist.append(avg_val)
        
        if epoch % 10 == 0:
            # print(f"Epoch {epoch}: Train MSE {avg_train:.4f} | Val MSE {avg_val:.4f}")
            print(f"Epoch {epoch}: Train MSE {avg_train:.8f} | Val MSE {avg_val:.8f}")

    return model, train_hist, val_hist