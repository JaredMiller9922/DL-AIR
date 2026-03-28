import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from utils.data_utils.dataset import SavedRFDataset
from utils.plot_utils.plotting_utils import BeautifulRFPlotter
from models import *
from utils.model_utils.losses import *

def make_loader(data_dir, batch_size=4, shuffle=False):
    ds = SavedRFDataset(data_dir)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return loader, ds

def validation_loss(model, val_dir="data/val", batch_size=16, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader, _ = make_loader(val_dir, batch_size=batch_size, shuffle=False)

    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            pred = model(x)
            loss = mse_loss(pred, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_examples += bs

    return total_loss / total_examples

def train(
    model,
    train_dir="data/train",
    val_dir="data/val",
    batch_size=16,
    epochs=300,
    lr=1e-3,
    weight_decay=0,
    device=None,
    save_path="model.pt",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader, ds = make_loader(train_dir, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_history = []
    val_history = []

    print(f"Device: {device}")
    print(f"Dataset size: {len(ds)}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Loss: {pit_mse_loss.__name__}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0
        for batch in loader:
            x, y, y_alt = batch["x"].to(device), batch["y"].to(device), batch["y_alt"].to(device)
            
            pred = model(x)
            # Use the PIT version!
            loss = pit_mse_loss(pred, y, y_alt) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train = epoch_train_loss / len(loader)
        avg_val = validation_loss(model, val_dir, device=device) # Update this to use PIT too
        
        train_history.append(avg_train)
        val_history.append(avg_val)

    torch.save(model.state_dict(), save_path)
    return model, val_history



def evaluate(
    model,
    model_path,
    val_dir="data/val",
    batch_size=16,
    device=None,
    plotter=None # <-- Add this argument
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader, ds = make_loader(val_dir, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_loss = 0.0
    total_examples = 0
    
    saved_plot = False # Flag to ensure we only plot one batch

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            pred = model(x)
            loss = pit_mse_loss(pred, y)
            
            # --- Trigger the plotting sequence on the first batch ---
            if plotter is not None and not saved_plot:
                model_name = model.__class__.__name__
                # Plot the raw pipeline (just once for the first model is fine, but safe to overwrite)
                plotter.plot_data_pipeline(x, y) 
                # Plot the overlay of True vs Predicted
                plotter.plot_separation_performance(y, pred, model_name=model_name)
                saved_plot = True
            # --------------------------------------------------------

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_examples += bs

    avg_loss = total_loss / total_examples
    print(f"Validation MSE loss: {avg_loss:.8f}")
    return avg_loss

# if __name__ == "__main__":
#     hybird_model = HybridSeparator(in_ch=8, out_ch=4)
#     linear_model = LinearSeparator(in_ch=8, out_ch=4)
#     tiny_model = TinySeparator(in_ch=8, hidden=64, out_ch=4)
#     lstm_model = LSTMSeparator(in_ch=8, out_ch=4)

#     hybrid_model, hybrid_val = train(
#         model=hybird_model,
#         train_dir="data/train",
#         val_dir="data/val",
#         epochs=50,
#         lr=1e-3,
#         save_path="hybrid_model.pt",
#     )

#     linear_model, linear_val = train(
#         model=linear_model,
#         train_dir="data/train",
#         val_dir="data/val",
#         epochs=50,
#         lr=1e-3,
#         save_path="linear_model.pt",
#     )
    
#     lstm_model, lstm_val = train(
#         model=lstm_model,
#         train_dir="data/train",
#         val_dir="data/val",
#         epochs=50,
#         lr=1e-3,
#         save_path="lstm_model.pt",
#     )

#     tiny_model, tiny_val = train(
#         model=tiny_model,
#         train_dir="data/train",
#         val_dir="data/val",
#         epochs=50,
#         lr=1e-3,
#         save_path="tiny_model.pt",
#     )




#     print("\n--- Validation ---")
    
#     # Initialize your visualizer
#     plotter = BeautifulRFPlotter(save_dir="plots")

#     evaluate(
#         model=HybridSeparator(in_ch=8, out_ch=4),
#         model_path="hybrid_model.pt",
#         val_dir="data/val",
#         plotter=plotter # <-- Pass it here
#     )

#     evaluate(
#         model=LinearSeparator(in_ch=8, out_ch=4),
#         model_path="linear_model.pt",
#         val_dir="data/val",
#         plotter=plotter
#     )

#     evaluate(
#         model=LSTMSeparator(in_ch=8, out_ch=4),
#         model_path="lstm_model.pt",
#         val_dir="data/val",
#         plotter=plotter
#     )


#     # evaluate(
#         # model=TinySeparator(in_ch=8, hidden=64, out_ch=4),
#         # model_path="tiny_model.pt",
#         # val_dir="data/val",
#     # )

# plt.plot(hybrid_val, label="HybridSeparator")
# plt.plot(linear_val, label="LinearSeparator")
# plt.plot(lstm_val, label="LSTMSeparator")
# # plt.plot(tiny_val, label="TinySeparator")
# plt.xlabel("Epoch")
# plt.ylabel("Validation MSE")
# plt.title("Validation MSE Across Epochs")
# plt.legend()
# plt.grid(True)

# plt.savefig("validation_curve.png", dpi=300)
# print("Saved plot to validation_curve.png")
# plt.close()