# Assignment 1: Neural Networks

## Key Terms

### Supervised Learning Task
A **supervised learning task** is like having a teacher guide your learning process. In this context, it means the neural network learns by being shown examples where both the input (like an image of a cat) and the correct output (the label "cat") are provided during training. The network uses these input-output pairs to learn patterns and make predictions on new, unseen data. This differs from unsupervised learning (where no correct answers are given) or reinforcement learning (where the network learns through trial and error with rewards/penalties).

### Training Time of Neural Networks (NN)
The **training time of a neural network** refers to how long it takes for the network to learn from the training data. This includes:
- The computational time needed to process all training examples
- Multiple passes through the dataset (called epochs) until the network reaches acceptable performance
- The time for the network to adjust its internal parameters (weights and biases) based on the training examples

Think of it like learning to ride a bicycle - training time is how long you need to practice before you can ride confidently without falling.

### Seeded Replicates
**Seeded replicates** are like running the same experiment multiple times with controlled randomness. When training neural networks, many processes involve randomness (like initial weight values, data shuffling, or dropout). A "seed" is a starting number that controls this randomness - think of it as setting the starting point for a random number generator.

By using the same seed across multiple training runs (replicates), you ensure that the randomness is identical each time. This allows researchers to:
- Verify that their results are consistent and reproducible
- Compare different methods fairly under identical conditions
- Distinguish between genuine performance differences and random variation

For example, if you train the same network 5 times with different seeds, you might get slightly different final accuracies due to random factors. Seeded replicates help you understand the typical performance range and reliability of your model.

### Network Hyperparameters
**Network hyperparameters** are the configuration settings that you choose before training begins - they're like the "recipe settings" for baking a cake. Unlike the network's weights (which the network learns during training), hyperparameters are decisions you make as the designer.

Common hyperparameters include:
- **Learning rate**: How big steps the network takes when adjusting its weights (like choosing how fast to walk while learning a new route)
- **Number of layers**: How deep your network is (like choosing how many floors your building will have)
- **Number of neurons per layer**: How wide each layer is (like choosing how many rooms per floor)
- **Batch size**: How many training examples to process at once (like choosing how many students to teach simultaneously)
- **Number of epochs**: How many times to go through the entire training dataset

These choices significantly impact how well and how quickly your network learns. Finding good hyperparameters often requires experimentation and is considered both an art and a science in machine learning.

### Loss Curves
**Loss curves** are visual plots that show how well a neural network is learning over time during training. Think of them as a "learning progress report" that tracks the network's mistakes as it improves.

#### What is "Lost"?
The term "loss" doesn't mean something is missing - it refers to the **error** or **mistake** the network makes when trying to predict the correct answer. Specifically:

- **Loss** measures how far off the network's predictions are from the true, correct answers
- A high loss means the network is making big mistakes (like guessing a cat is a dog with high confidence)
- A low loss means the network's predictions are very close to the correct answers
- The goal of training is to minimize (reduce) this loss over time

Think of loss like the score in golf - lower numbers are better, and you want to minimize your "mistakes."

#### Axes of Loss Curve Plots
Loss curves typically have two axes:

**X-axis (Horizontal)**: **Time progression during training**
- Often measured in **epochs** (complete passes through the training dataset)
- Sometimes measured in **iterations** or **batches** (smaller chunks of training)
- Could also be measured in actual **wall-clock time** (minutes/hours)
- This shows the progression of training from start to finish

**Y-axis (Vertical)**: **Loss value** (the magnitude of error)
- Shows the numerical value of the loss function (like cross-entropy loss or mean squared error)
- Higher values = worse performance (more mistakes)
- Lower values = better performance (fewer mistakes)
- The scale depends on the specific loss function used

#### Typical Loss Curve Patterns
A healthy loss curve usually:
- Starts high (network makes many mistakes initially)
- Decreases over time (network learns and improves)
- Eventually plateaus (network reaches its learning limit)
- May show some fluctuation (normal variation in learning process)

Loss curves help you understand if your network is learning properly, learning too slowly, or encountering problems like overfitting.

### Swept Volume in Robotics Context

In robotics, the **swept volume** refers to the total 3D space that a robot occupies as it moves along a path from one configuration to another. Think of it like the "shadow" or "trail" that the robot leaves behind as it moves through space.

#### Understanding Swept Volume
Imagine you're painting a fence with a paintbrush:
- The **paintbrush** represents the robot
- The **stroke you make** represents the robot's path
- The **painted area on the fence** represents the swept volume

More technically:
- **Swept volume** = The union of all spaces occupied by the robot at every point along its trajectory
- It's the 3D region that gets "swept out" or "traced" by the robot's body as it moves
- This includes the robot's links, joints, and any attached tools or end-effectors

#### Why Swept Volume Matters
Calculating swept volume is crucial for:

1. **Collision Detection**: Ensuring the robot doesn't hit obstacles during movement
2. **Path Planning**: Finding safe routes that avoid collisions with the environment
3. **Safety Analysis**: Determining which areas humans should avoid when the robot is operating
4. **Workspace Analysis**: Understanding what regions the robot can safely access

#### Key Characteristics
- **Dynamic**: Changes based on the specific path the robot takes
- **3D Shape**: Usually an irregular, complex geometric shape
- **Path-Dependent**: Different paths between the same start and end points create different swept volumes
- **Conservative Estimate**: Often calculated as a slightly larger volume than actual to ensure safety margins

#### Practical Applications
- **Manufacturing**: Ensuring robotic arms don't collide with assembly line equipment
- **Surgery**: Planning robot-assisted surgical procedures to avoid critical anatomy
- **Autonomous Vehicles**: Path planning that considers the vehicle's full body dimensions
- **Space Missions**: Planning robot movements in cramped spacecraft environments

The swept volume calculation is essential for safe and efficient robot operation in any environment with obstacles or space constraints.

- setup a virtual env for CPU, instructions in assignment
