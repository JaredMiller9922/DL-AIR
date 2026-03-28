import torch
from utils.model_utils.losses import pit_mse_loss

def train_model(model, train_loader, val_loader, epochs=50, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_hist, val_hist = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0
        for batch in train_loader:
            x, y, y_alt = batch["x"].to(device), batch["y"].to(device), batch["y_alt"].to(device)
            pred = model(x)
            loss = pit_mse_loss(pred, y, y_alt)
            
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
                vx, vy, vy_alt = v_batch["x"].to(device), v_batch["y"].to(device), v_batch["y_alt"].to(device)
                v_pred = model(vx)
                total_val += pit_mse_loss(v_pred, vy, vy_alt).item()
        
        avg_val = total_val / len(val_loader)
        train_hist.append(avg_train)
        val_hist.append(avg_val)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train MSE {avg_train:.4f} | Val MSE {avg_val:.4f}")

    return model, train_hist, val_hist