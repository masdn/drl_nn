import numpy as np
import torch as t
from torch.utils.data import DataLoader, TensorDataset
import time

#TODO Check over all parameters tested and ensure AI didn't hallucinate
#     There may be more parameters we need to test. 
#TODO Self-Directed Investigation portion of assignment, need hypothesis first about it.

def assignment():
    return "cpu"


class NeuralNetwork(t.nn.Module):
    """
    A simple feedforward neural network for swept volume prediction.
    
    This network consists of:
    - An input layer that accepts features
    - Hidden layers with ReLU activation functions
    - An output layer for predictions
    """
    
    def __init__(self, in_dimension, out_dimension, hidden_layers, neurons_per_hidden_layer = 64):
        """
        Initialize the neural network.
        
        Args:
            in_dimension (int): Number of input features
            out_dimension (int): Number of output predictions
            hidden_layers (int): Number of hidden layers
            neurons_per_hidden_layer (int): Number of neurons per hidden layer (default 64)
        """
        super(NeuralNetwork, self).__init__()
        
        # Create a list to store all layers
        layers = []
        
        # Add first hidden layer (input -> first hidden)
        current_size = in_dimension
        for _ in range(hidden_layers):
            layers.append(t.nn.Linear(current_size, neurons_per_hidden_layer))
            layers.append(t.nn.ReLU())  # Activation function
            #layers.append(t.nn.LeakyReLU()) #TODO try this, it lowers the loss by ~6 points
            current_size = neurons_per_hidden_layer
        
        # Add output layer (last hidden -> output)
        layers.append(t.nn.Linear(current_size, out_dimension))
        
        # Initialize weights and biases for better training performance
        for layer in layers:
            if isinstance(layer, t.nn.Linear):
                # Xavier/Glorot initialization for weights
                # This helps prevent vanishing/exploding gradients
                t.nn.init.xavier_uniform_(layer.weight)
                # Initialize biases to zero
                t.nn.init.zeros_(layer.bias)


        # Combine all layers into a sequential model
        self.network = t.nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Network predictions
        """
        
        
        return self.network(x)


# Create and instantiate the neural network
# This demonstrates how to create a neural network with the class we just defined

def create_neural_network(data, hidden_layers=1, neurons_per_hidden_layer=64):
    """
    Create and return a neural network instance.
    This function demonstrates how to instantiate the SimpleNeuralNetwork class.
    
    Returns:
        SimpleNeuralNetwork: An initialized neural network ready for training
    """
    # Define network architecture parameters
    in_dimension = data["training_features"].shape[1]    # Example: 28x28 pixel images flattened (like MNIST digits)
    out_dimension = data["training_labels"].shape[1]       # Example: 10 classes for digit classification (0-9)
    # Create the neural network instance
    model = NeuralNetwork(
        in_dimension=in_dimension,
        out_dimension=out_dimension,
        hidden_layers=hidden_layers,
        neurons_per_hidden_layer=neurons_per_hidden_layer
    )
    return model



def setup_training_components(model, optimizer_type="adam", learning_rate=0.01):
    """
    Set up the loss function and optimizer for training the neural network.
    
    Args:
        model (SimpleNeuralNetwork): The neural network to train
        learning_rate (float): Controls how big steps the optimizer takes during learning
                              (smaller = more careful learning, larger = faster but riskier)
    
    Returns:
        tuple: (loss_function, optimizer) ready for training
    """
    
    # Loss Function: CrossEntropyLoss
    # This measures how "wrong" our predictions are compared to the true labels
    # Think of it like grading a multiple-choice test - it penalizes confident wrong answers more
    #loss_function = t.nn.CrossEntropyLoss()
    loss_function = t.nn.MSELoss()
    
    # Optimizer: Adam
    # This is the "learning algorithm" that adjusts the network's weights to reduce loss
    # Adam is popular because it adapts the learning rate automatically and works well in practice

    if optimizer_type == "adam":
        optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = t.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == "rmsprop":
        optimizer = t.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    return loss_function, optimizer

def train_and_evaluate(data, hidden_layers, neurons_per_hidden_layer, optimizer_type, learning_rate, train_size, batch_size=1000, num_epochs=1000):
    """
    Train a neural network with the given configuration and evaluate it.

    Args:
        data (dict): Dictionary of tensors with training, evaluation, and testing features/labels.
        hidden_layers (int): Number of hidden layers in the network.
        neurons_per_hidden_layer (int): Number of neurons in each hidden layer.
        optimizer_type (str): Optimizer type ("adam", "sgd", "rmsprop").
        learning_rate (float): Learning rate for the optimizer.
        train_size (int): Number of samples to use for training (subsample if smaller than dataset).
        batch_size (int): Batch size for training (default: 1000).
        num_epochs (int): Number of training epochs (default: 1000).

    Saves:
        - Best model weights (`best_model_*.pth`) when validation loss improves.
        - Loss histories and configuration (`losses_*.npz`) for later analysis.
        - Training time in seconds.
    """
    # Track training start time
    start_time = time.time()

    # Subsample the training data if a smaller train_size is requested
    if train_size < len(data["training_features"]):
        indices = t.randperm(len(data["training_features"]))[:train_size]
        train_features = data["training_features"][indices]
        train_labels = data["training_labels"][indices]
    else:
        train_features = data["training_features"]
        train_labels = data["training_labels"]

    # Wrap features and labels in a TensorDataset and DataLoader for batching
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Build a fresh neural network and optimizer/loss function for this run
    neural_network = create_neural_network(data, hidden_layers, neurons_per_hidden_layer)
    loss_fn, optimizer = setup_training_components(neural_network, optimizer_type, learning_rate)

    # Track losses across epochs
    training_loss_history, validation_loss_history, test_loss_history = [], [], []
    best_val_loss = float("inf")
    best_model_state = None

    # Main training loop
    for epoch in range(num_epochs):
        neural_network.train()
        epoch_loss = 0.0

        # Train over all mini-batches
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            predictions = neural_network(batch_features)
            loss = loss_fn(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_features.size(0)
        epoch_loss /= len(train_loader.dataset)

        # Validation step
        neural_network.eval()
        with t.no_grad():
            val_predictions = neural_network(data["evaluation_features"])
            val_loss = loss_fn(val_predictions, data["evaluation_labels"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = neural_network.state_dict()
                t.save(best_model_state, f"best_model_h{hidden_layers}_n{neurons_per_hidden_layer}_opt{optimizer_type}_lr{learning_rate}_ts{train_size}_seed{t.initial_seed()}.pth")

        # Test step
        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            with t.no_grad():
                test_predictions = neural_network(data["testing_features"])
                test_loss = loss_fn(test_predictions, data["testing_labels"])
                test_loss_history.append(test_loss.item())

        training_loss_history.append(epoch_loss)
        validation_loss_history.append(val_loss.item())

    # Compute training time
    end_time = time.time()
    train_time = end_time - start_time

    # Save losses, config, and training time
    np.savez(
        f"losses_h{hidden_layers}_n{neurons_per_hidden_layer}_opt{optimizer_type}_lr{learning_rate}_ts{train_size}_seed{t.initial_seed()}.npz",
        training_loss=np.array(training_loss_history),
        validation_loss=np.array(validation_loss_history),
        test_loss=np.array(test_loss_history),
        config=dict(
            hidden_layers=hidden_layers,
            neurons_per_hidden_layer=neurons_per_hidden_layer,
            optimizer=optimizer_type,
            learning_rate=learning_rate,
            train_size=train_size,
            batch_size=batch_size,
            seed=t.initial_seed(),
            train_time=train_time,
        ),
    )

def main():
    device = "cpu"
    data = dict(np.load("swept_volume_data.npz"))
    for key, value in data.items():
        data[key] = t.tensor(value, dtype=t.float32, device=device)

    hidden_layers_options = [1]
    neurons_options = [64]
    optimizers = ["adam"]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    train_sizes = [1000, 10000, 100000]

    # List of seeds for replicates
    seeds = [13, 42, 99, 123, 2025]

    for hl in hidden_layers_options:
        for n in neurons_options:
            for opt in optimizers:
                for lr in learning_rates:
                    for ts in train_sizes:
                        for seed in seeds:
                            print(f"Running config: hidden_layers={hl}, neurons={n}, optimizer={opt}, lr={lr}, train_size={ts}, seed={seed}")
                            t.manual_seed(seed)
                            train_and_evaluate(data, hl, n, opt, lr, ts)


if __name__ == "__main__":    
    main()
