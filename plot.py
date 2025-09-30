import numpy as np
import matplotlib.pyplot as plt
import torch as t
import time
import pandas as pd
import glob
import os

# Collect all saved loss files
loss_files = glob.glob("losses_*.npz")

results = []
timings = []


# --- Plot and save evaluation loss curves for each setup ---
from collections import defaultdict
plot_dir = "loss_plots"
os.makedirs(plot_dir, exist_ok=True)

# Group files by (learning_rate, train_size, seed) for individual plots
grouped_files = defaultdict(list)

for file in loss_files:
    data = np.load(file, allow_pickle=True)
    config = data["config"].item()

    training_loss = data["training_loss"]
    validation_loss = data["validation_loss"]
    test_loss = data["test_loss"]

    # Epoch indices
    epochs = np.arange(1, len(training_loss) + 1)
    val_epochs = epochs
    if len(test_loss) < len(training_loss) and len(test_loss) > 0:
        step = max(1, len(training_loss) // len(test_loss))
        test_epochs = np.arange(step, step * len(test_loss) + 1, step)
    else:
        test_epochs = np.arange(1, len(test_loss) + 1)

    # Find best validation epoch
    best_val_epoch = np.argmin(validation_loss) + 1
    best_val_loss = validation_loss[best_val_epoch - 1]

    # Match closest test loss to best validation epoch
    if len(test_loss) > 0:
        closest_idx = np.argmin(np.abs(test_epochs - best_val_epoch))
        test_loss_at_best = test_loss[closest_idx]
    else:
        test_loss_at_best = None

    results.append({
        "hidden_layers": config["hidden_layers"],
        "neurons": config["neurons_per_hidden_layer"],
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
        "train_size": config["train_size"],
        "batch_size": config["batch_size"],
        "seed": config["seed"],
        "best_val_epoch": best_val_epoch,
        "best_val_loss": best_val_loss,
        "test_loss_at_best": test_loss_at_best,
    })

    # If training time tracking is added, append it here
    if "train_time" in config:
        timings.append({
            "hidden_layers": config["hidden_layers"],
            "neurons": config["neurons_per_hidden_layer"],
            "optimizer": config["optimizer"],
            "learning_rate": config["learning_rate"],
            "train_size": config["train_size"],
            "seed": config["seed"],
            "train_time": config["train_time"]
        })

    # --- Plot and save for this run ---
    setup_str = f"hl{config['hidden_layers']}_n{config['neurons_per_hidden_layer']}_opt{config['optimizer']}_lr{config['learning_rate']}_ts{config['train_size']}_seed{config['seed']}"
    plt.figure(figsize=(8,5))
    plt.plot(epochs, training_loss, label="Training Loss")
    plt.plot(epochs, validation_loss, label="Validation Loss")
    if len(test_loss) > 0:
        plt.plot(test_epochs, test_loss, label="Test Loss")
    plt.axvline(best_val_epoch, color='r', linestyle='--', label=f"Best Val Epoch: {best_val_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves: {setup_str}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"loss_curve_{setup_str}.png"))
    plt.close()

    # Group for aggregate plots
    key = (config["learning_rate"], config["train_size"], config["optimizer"])
    grouped_files[key].append((epochs, validation_loss, config["seed"]))

# Convert results to DataFrame

results_df = pd.DataFrame(results)
test_loss_table = results_df.groupby([
    "hidden_layers", "neurons", "optimizer", "learning_rate", "train_size"
])["test_loss_at_best"].agg(["mean", "std", "min", "max"])

# Save the test loss table as CSV for report inclusion
test_loss_table.to_csv("test_loss_table.csv")

# Optionally, print a message to confirm
print("Test loss table saved as test_loss_table.csv")

# If timings were tracked, show mean ± std
if timings:
    timings_df = pd.DataFrame(timings)
    print("\n=== Training Times (seconds) ===")
    print(timings_df.groupby(["hidden_layers", "neurons", "optimizer", "learning_rate", "train_size"])[
        "train_time"
    ].agg(["mean", "std"]))


# --- Aggregate plots for each (learning_rate, train_size, optimizer) ---
for key, runs in grouped_files.items():
    lr, ts, opt = key
    plt.figure(figsize=(8,5))
    for epochs, val_loss, seed in runs:
        plt.plot(epochs, val_loss, label=f"seed={seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(f"Validation Loss Curves: lr={lr}, train_size={ts}, opt={opt}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"val_loss_agg_lr{lr}_ts{ts}_opt{opt}.png"))
    plt.close()


# --- Compare learning rates for same train_size, optimizer, and seed ---
compare_lr = True
if compare_lr:
    from collections import defaultdict
    lr_compare_groups = defaultdict(list)
    for file in loss_files:
        data = np.load(file, allow_pickle=True)
        config = data["config"].item()
        val_loss = data["validation_loss"]
        epochs = np.arange(1, len(val_loss) + 1)
        key = (config["train_size"], config["optimizer"], config["seed"])
        lr_compare_groups[key].append((config["learning_rate"], epochs, val_loss))
    for (ts, opt, seed), runs in lr_compare_groups.items():
        plt.figure(figsize=(8,5))
        for lr, epochs, val_loss in sorted(runs):
            plt.plot(epochs, val_loss, label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title(f"Validation Loss: train_size={ts}, opt={opt}, seed={seed}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"val_loss_compare_lr_ts{ts}_opt{opt}_seed{seed}.png"))
        plt.close()

# --- Compare training sizes for same learning rate, optimizer, and seed ---
compare_ts = True
if compare_ts:
    ts_compare_groups = defaultdict(list)
    for file in loss_files:
        data = np.load(file, allow_pickle=True)
        config = data["config"].item()
        val_loss = data["validation_loss"]
        epochs = np.arange(1, len(val_loss) + 1)
        key = (config["learning_rate"], config["optimizer"], config["seed"])
        ts_compare_groups[key].append((config["train_size"], epochs, val_loss))
    for (lr, opt, seed), runs in ts_compare_groups.items():
        plt.figure(figsize=(8,5))
        for ts, epochs, val_loss in sorted(runs):
            plt.plot(epochs, val_loss, label=f"train_size={ts}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title(f"Validation Loss: lr={lr}, opt={opt}, seed={seed}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"val_loss_compare_ts_lr{lr}_opt{opt}_seed{seed}.png"))
        plt.close()
