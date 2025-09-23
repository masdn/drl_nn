import numpy as np
import matplotlib.pyplot as plt

data = np.load("losses.npz")

training_loss = data["training_loss"]
validation_loss = data["validation_loss"]
evaluation_loss = data["evaluation_loss"]

plt.plot(training_loss, label="Training Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.plot(evaluation_loss, label="Evaluation Loss") #TODO fix the issue with the evaluation loss

plt.legend()
plt.show()