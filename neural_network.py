import numpy as np
import torch as t

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
    
    def __init__(self, in_dimension, hidden_layers, output_size, neurons_per_hidden_layer = 64):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Number of input features
            hidden_sizes (list): List of hidden layer sizes [e.g., [128, 64, 32]]
            output_size (int): Number of output predictions
            neurons_per_hidden_layer (int): Number of neurons per hidden layer Default is 64
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
        layers.append(t.nn.Linear(current_size, output_size))
        
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

def create_neural_network(data):
    """
    Create and return a neural network instance.
    This function demonstrates how to instantiate the SimpleNeuralNetwork class.
    
    Returns:
        SimpleNeuralNetwork: An initialized neural network ready for training
    """
    # Define network architecture parameters
    input_size = data["training_features"].shape[1]    # Example: 28x28 pixel images flattened (like MNIST digits)
    hidden_layers = 1  # Two hidden layers with decreasing sizes
    output_size = data["training_labels"].shape[1]       # Example: 10 classes for digit classification (0-9)
    
    # Create the neural network instance
    model = NeuralNetwork(
        in_dimension=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,

    )
    
    return model



def setup_training_components(model, learning_rate=0.01):
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
    optimizer = t.optim.Adam(
        model.parameters(),  # Tell optimizer which weights to update (all network parameters)
        lr=learning_rate     # Learning rate controls step size during weight updates
    )
    
    return loss_function, optimizer


def main():
    t.manual_seed(13)

    device = "cuda" if t.cuda.is_available() else "cpu"
    data = dict(np.load("swept_volume_data.npz"))

    for key, value in data.items():
        print(key, value.shape)
        data[key] = t.tensor(value, dtype=t.float32, device=device)
    # Instantiate the neural network
    neural_network = create_neural_network(data)

    # Print network architecture for verification
    print("Neural Network Architecture:")
    print(f"Input size: {neural_network.network[0].in_features}")
    print(f"Hidden layers: {[layer.out_features for layer in neural_network.network if isinstance(layer, t.nn.Linear)][:-1]}")
    print(f"Output size: {neural_network.network[-1].out_features}")
    print("\nFull network structure:")
    print(neural_network)

    # Define loss function and optimizer
    # These are essential components for training the neural network
        
    # Create loss function and optimizer for our neural network
    loss_fn, optimizer = setup_training_components(neural_network)

    print("\nTraining Components Setup:")
    print(f"Loss function: {loss_fn}")
    print(f"Optimizer: {optimizer}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")


    num_epochs = 1000
    for epoch in range(num_epochs):
        # Training phase
        neural_network.train()  # Set network to training mode
        optimizer.zero_grad()
        predictions = neural_network(data["training_features"])
        loss = loss_fn(predictions, data["training_labels"])
        loss.backward() #does the backpropagation
        optimizer.step()
        
        # Validation phase
        # Track best validation loss and save model when it improves
        if epoch == 0:
            best_val_loss = float('inf')  # Initialize with infinity for first comparison
            best_model_state = None

        neural_network.eval()  # Set network to evaluation mode
        with t.no_grad():  # Disable gradient computation for validation
            val_predictions = neural_network(data["evaluation_features"])
            val_loss = loss_fn(val_predictions, data["evaluation_labels"])
            print(f"Validation Loss: {val_loss.item():.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = neural_network.state_dict()
                print(f"New best validation loss: {best_val_loss:.4f}")
                print("Saving model...")
                t.save(best_model_state, "best_model.pth")

        # Testing phase - evaluate on test data every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            with t.no_grad():  # Disable gradient computation for testing
                test_predictions = neural_network(data["testing_features"])
                test_loss = loss_fn(test_predictions, data["testing_labels"])
                print(f"Test Loss: {test_loss.item():.4f}")

        # Initialize loss history lists before the training loop (only on first epoch)
        if epoch == 0:
            training_loss_history = []
            validation_loss_history = []
            test_loss_history = []
        
        # Store current epoch losses in history lists
        training_loss_history.append(loss.item())
        validation_loss_history.append(val_loss.item())
        
        # Add test loss to history when it's computed (every 100 epochs or last epoch)
        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            test_loss_history.append(test_loss.item())
        
        # Save all loss histories to npz file on the final epoch
        if epoch == num_epochs - 1:
            np.savez("losses.npz", 
                    training_loss=np.array(training_loss_history),
                    validation_loss=np.array(validation_loss_history), 
                    evaluation_loss=np.array(test_loss_history))
            print("Loss histories saved to losses.npz")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
    return "cpu"

if __name__ == "__main__":    
    main()