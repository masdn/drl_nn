import numpy as np
import matplotlib.pyplot as plt

data = np.load("losses.npz")

training_loss = data["training_loss"]
validation_loss = data["validation_loss"]
evaluation_loss = data["evaluation_loss"]

# Build epoch indices for each series so they align on the x-axis
epochs = np.arange(1, len(training_loss) + 1)
val_epochs = epochs

# Evaluation (test) loss was recorded less frequently (every N epochs). Infer N.
if len(evaluation_loss) < len(training_loss) and len(evaluation_loss) > 0:
    step = max(1, len(training_loss) // len(evaluation_loss))
    eval_epochs = np.arange(step, step * len(evaluation_loss) + 1, step)
else:
    eval_epochs = np.arange(1, len(evaluation_loss) + 1)

plt.plot(epochs, training_loss, label="Training Loss")
plt.plot(val_epochs, validation_loss, label="Validation Loss")
plt.plot(eval_epochs, evaluation_loss, label="Evaluation Loss")

plt.legend()
plt.show()