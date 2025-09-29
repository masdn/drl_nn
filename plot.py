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

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\n=== Test Losses at Best Validation Epoch ===")
print(results_df.groupby(["hidden_layers", "neurons", "optimizer", "learning_rate", "train_size"])[
    "test_loss_at_best"
].agg(["mean", "std", "min", "max"]))

# If timings were tracked, show mean ± std
if timings:
    timings_df = pd.DataFrame(timings)
    print("\n=== Training Times (seconds) ===")
    print(timings_df.groupby(["hidden_layers", "neurons", "optimizer", "learning_rate", "train_size"])[
        "train_time"
    ].agg(["mean", "std"]))

# Example plot for just one run (optional visualization)
example_file = loss_files[0]
data = np.load(example_file, allow_pickle=True)
plt.plot(data["training_loss"], label="Training Loss")
plt.plot(data["validation_loss"], label="Validation Loss")
if len(data["test_loss"]) > 0:
    step = max(1, len(data["training_loss"]) // len(data["test_loss"]))
    test_epochs = np.arange(step, step * len(data["test_loss"]) + 1, step)
    plt.plot(test_epochs, data["test_loss"], label="Test Loss")
plt.legend()
plt.title(f"Example run: {os.path.basename(example_file)}")
plt.show()
