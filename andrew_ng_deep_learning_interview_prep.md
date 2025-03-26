# Andrew Ng's Deep Learning Concepts

## 1. Neural Networks Fundamentals

### 1.1 Basic Neural Network Concepts

#### Single Neuron Architecture
- **Mathematical Model**: `y = σ(wx + b)` where σ is an activation function
- **Components**:
  - Inputs (x): Features fed into the network
  - Weights (w): Parameters that determine feature importance
  - Bias (b): Allows shifting the activation function
  - Activation Function (σ): Introduces non-linearity
- **Comparison to Linear Regression**: Linear regression without activation function vs. neural networks with non-linear activation functions

#### Deep Network Structure
- **Layer Types**:
  - Input Layer: Raw data features (dimension equals feature count)
  - Hidden Layers: Extract progressively more abstract features
  - Output Layer: Produces final prediction or classification
- **Learning Process**:
  - Lower layers learn simple patterns (edges, colors)
  - Middle layers learn compositions (shapes, textures)
  - Higher layers learn complex concepts (objects, scenes)
- **Representation**: For L-layer network, parameters include {W¹, b¹, W², b², ..., Wᴸ, bᴸ} with activations {a⁰, a¹, a²,..., aᴸ}

### 1.2 Neural Network Applications

#### Computer Vision (CNNs)
- **Architecture Features**:
  - Convolutional layers: Extract spatial features using filters
  - Pooling layers: Downsample and create position invariance
  - Fully connected layers: Final classification based on extracted features
- **Applications**:
  - Image classification
  - Object detection
  - Image segmentation
  - Style transfer

#### Natural Language Processing (RNNs)
- **Architecture Features**:
  - Sequential processing of data
  - Memory cells that retain information over sequences
  - Variants like LSTM and GRU to handle long-term dependencies
- **Applications**:
  - Text generation
  - Machine translation
  - Sentiment analysis
  - Speech recognition

#### Structured Data (Standard NNs)
- **Architecture Features**:
  - Fully connected networks
  - Custom architectures for tabular data
- **Applications**:
  - Recommendation systems
  - Financial predictions
  - Risk assessment
  - Customer segmentation

### 1.3 Components of Neural Networks

#### Activation Functions
- **Sigmoid**: `σ(z) = 1/(1+e^(-z))`
  - Range: [0,1]
  - Use case: Binary classification output layers
  - Limitation: Vanishing gradient for extreme values
  
- **Tanh**: `tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))`
  - Range: [-1,1]
  - Advantage: Zero-centered outputs, helps with next layer learning
  - Limitation: Still has vanishing gradient problem
  
- **ReLU (Rectified Linear Unit)**: `ReLU(z) = max(0,z)`
  - Range: [0,∞)
  - Advantage: No vanishing gradient for positive values, computational efficiency
  - Limitation: "Dying ReLU" problem when neurons consistently output 0
  
- **Leaky ReLU**: `LeakyReLU(z) = max(0.01z, z)`
  - Advantage: Prevents dying neurons by allowing small gradient when z < 0
  - Variants: Parametric ReLU allows learning the slope parameter

#### Forward/Backward Propagation
- **Forward Propagation Equations**:
  - `Z[l] = W[l] · A[l-1] + b[l]`
  - `A[l] = g[l](Z[l])` where g is the activation function
  
- **Backward Propagation Equations**:
  - `dZ[L] = A[L] - Y` (for binary classification with sigmoid)
  - `dW[l] = (1/m) · dZ[l] · A[l-1].T`
  - `db[l] = (1/m) · sum(dZ[l])`
  - `dZ[l-1] = W[l].T · dZ[l] * g'[l-1](Z[l-1])` (element-wise multiplication)
  
- **Gradient Descent Update**:
  - `W[l] := W[l] - α · dW[l]`
  - `b[l] := b[l] - α · db[l]`
  - α is the learning rate

#### Hyperparameters and Optimization
- **Critical Hyperparameters**:
  - Learning rate: Controls step size during gradient descent
  - Layer count: Determines network depth
  - Units per layer: Determines network width
  - Activation functions: Introduces non-linearity
  - Batch size: Number of examples processed before weight update
  
- **Optimization Algorithms**:
  - Gradient Descent: Basic optimization method
  - Momentum: Adds "velocity" to navigate ravines
  - RMSprop: Adaptive learning rates per parameter
  - Adam: Combines benefits of momentum and RMSprop

## 2. ML Project Structuring

### 2.1 Evaluation Strategy

#### Single Number Evaluation Metrics
- **Classification Metrics**:
  - Accuracy: `(TP+TN)/(TP+TN+FP+FN)` - Overall correctness
  - Precision: `TP/(TP+FP)` - Accuracy of positive predictions
  - Recall: `TP/(TP+FN)` - Coverage of actual positives
  - F1 Score: `2 * (precision * recall)/(precision + recall)` - Harmonic mean of precision and recall
  
- **Regression Metrics**:
  - Mean Squared Error: `(1/m) * Σ(y_pred - y_true)²`
  - Root Mean Squared Error: `sqrt(MSE)`
  - Mean Absolute Error: `(1/m) * Σ|y_pred - y_true|`

- **Optimizing vs. Satisficing Metrics**:
  - Optimizing metric: The main metric to improve (e.g., accuracy)
  - Satisficing metrics: Secondary metrics with minimum acceptable thresholds (e.g., inference time)
  - Example: "Maximize F1 score subject to inference time < 100ms"

#### Train/Dev/Test Splits
- **Traditional Splits**:
  - 70/30 (train/test) or 60/20/20 (train/dev/test) for small datasets
  
- **Modern Splits for Big Data**:
  - 98/1/1 or 99.5/0.25/0.25 for large datasets
  - Ensures sufficient validation while maximizing training data
  
- **Distribution Considerations**:
  - Dev and test sets must come from the same distribution
  - Train set can be from a slightly different distribution
  - Dev/test should reflect data you care about in production

### 2.2 Error Analysis Frameworks

#### Human-level Performance
- **Bayes Error Estimation**:
  - Human-level error as proxy for Bayes error rate (theoretical minimum)
  - Multiple human-level performance metrics possible (expert vs. average human)
  
- **Error Decomposition**:
  - Avoidable bias = Training error - Human-level error
  - Variance = Dev error - Training error
  - Data mismatch = Dev error - Train-dev error (when distributions differ)
  
- **Strategy Selection Based on Analysis**:
  - If avoidable bias is high: Focus on model capacity or training algorithm
  - If variance is high: Focus on regularization or more data
  - If data mismatch is high: Focus on making training data more similar to test data

#### Structured Error Analysis
- **Manual Error Categorization**:
  - Randomly sample misclassified examples (e.g., 100 errors)
  - Create taxonomy of error types
  - Count frequency of each error type
  
- **Error Analysis Spreadsheet**:
  - Columns: Error types (e.g., blurry images, incorrect labels)
  - Rows: Individual examples
  - Cells: Binary indicators for each error type
  - Summary: Percentage of errors in each category
  
- **Prioritization Framework**:
  - Calculate "ceiling" for improvement for each category
  - Estimate effort required for each improvement direction
  - Prioritize based on impact/effort ratio

### 2.3 Data Handling Strategies

#### Mismatched Distributions
- **Problem Identification**:
  - Train-dev set: Subset of training data held out, same distribution as training
  - Compare: Training error vs. train-dev error vs. dev error
  
- **Distribution Types**:
  - Training distribution: What your model trains on
  - Test distribution: What your model will encounter in production
  
- **Handling Strategies**:
  - Data augmentation to make training more like testing
  - Domain adaptation techniques
  - Collect more training data from test distribution

#### Transfer Learning
- **Implementation Approaches**:
  - Pre-training: Train on large dataset (e.g., ImageNet)
  - Fine-tuning options:
    1. Train only the final layer (feature extraction)
    2. Train the entire network with a small learning rate (full fine-tuning)
  
- **When Transfer Learning Works Best**:
  - Tasks A and B have the same input type
  - You have much more data for task A than task B
  - Low-level features from A are useful for B
  
- **Practical Steps**:
  1. Remove the last layer and its weights
  2. Replace with new layer for target task
  3. Either freeze pre-trained weights or allow them to update
  4. Train with task-specific data

#### Data Augmentation
- **Common Techniques**:
  - Images: Rotation, flipping, cropping, color shifts
  - Text: Synonym replacement, back-translation
  - Audio: Adding noise, pitch shifting, time stretching
  
- **Artificial Data Synthesis**:
  - Generate synthetic examples to address specific weaknesses
  - Example: Combine clean audio with background noise samples
  - Example: Use 3D rendering for additional training images
  
- **Considerations**:
  - Ensure augmentations create realistic examples
  - Avoid creating a biased sample of the space (avoid augmentation overfitting)
  - Validate effectiveness through cross-validation

## 3. Visualization Relevant Concepts

### 3.1 Model Performance Visualization

#### Learning Curves
- **Training vs. Validation Error Curves**:
  - X-axis: Number of training examples or epochs
  - Y-axis: Error metric
  - High bias (underfitting): Both errors plateau at similar, high values
  - High variance (overfitting): Large gap between training and validation errors
  
- **Learning Rate Analysis**:
  - Plot loss vs. iterations for different learning rates
  - Too high: Erratic or increasing loss
  - Too low: Slow convergence
  - Optimal: Steady decrease in loss
  
- **Regularization Effect Visualization**:
  - Plot training/validation error vs. regularization strength
  - Visualize the "sweet spot" where validation error is minimized

#### Error Breakdown
- **Confusion Matrix Visualization**:
  - Heat map showing prediction vs. actual classes
  - Highlights specific class confusion patterns
  
- **Error Distribution**:
  - Histograms of error magnitudes
  - Scatter plots of predicted vs. actual values
  - Error concentration in certain data regions
  
- **Error Evolution**:
  - Track error metrics over time/iterations
  - Visualize improvement in specific error categories

### 3.2 Feature Visualization

#### Activation Visualization
- **Layer Activations**:
  - Visualize activations for specific inputs
  - Compare activations across different examples
  - Track how activations evolve during training
  
- **Feature Maps**:
  - For CNNs, visualize the feature maps at each layer
  - Early layers: Edge detectors, textures
  - Later layers: Object parts, complete objects
  
- **Neuron Maximization**:
  - Generate inputs that maximize specific neuron activations
  - Reveals patterns that specific neurons are detecting

#### Dimensionality Reduction
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
  - Projects high-dimensional data to 2D/3D
  - Preserves local structure, good for visualization
  - Parameters: Perplexity controls neighborhood size
  
- **PCA (Principal Component Analysis)**:
  - Linear dimensionality reduction
  - Identifies directions of maximum variance
  - Useful for visualizing feature importance
  
- **UMAP (Uniform Manifold Approximation and Projection)**:
  - Newer technique with better global structure preservation
  - Often faster than t-SNE for large datasets

### 3.3 End-to-End vs. Pipeline Approaches

#### End-to-End Deep Learning
- **Direct Mapping Approach**:
  - Single network from raw input to desired output
  - Examples:
    - Audio → Text transcription
    - Image → Caption
    - Text → Translation
  
- **Visualization Challenges**:
  - "Black box" nature requires specialized visualization
  - Attention mechanisms can provide insights into focus areas
  - Gradient-based methods to identify important input regions

#### Pipeline Visualization
- **Component Performance Metrics**:
  - Visualize accuracy/error at each pipeline stage
  - Identify bottlenecks in multi-stage systems
  
- **Error Propagation**:
  - Track how errors cascade through pipeline stages
  - Visualize confidence scores between components
  
- **Integration Visualization**:
  - Show data transformations between pipeline stages
  - Highlight information loss or gain at each transition

## 4. Key Optimization Techniques

### 4.1 Improving Model Performance

#### Bias Reduction Strategies
- **Model Capacity Increase**:
  - Add more layers (deeper networks)
  - Add more units per layer (wider networks)
  - Use more complex architectures (transformers, etc.)
  
- **Advanced Optimization**:
  - Learning rate scheduling (decay, warm-up)
  - Use more advanced optimizers (Adam, AdamW)
  - Batch normalization, layer normalization
  
- **Architecture Refinement**:
  - Residual connections (ResNet-style)
  - Dense connections (DenseNet-style)
  - Attention mechanisms

#### Variance Reduction Strategies
- **Data Augmentation Implementation**:
  - Online augmentation during training
  - Multiple augmentations per sample
  - Adaptive augmentation based on model weaknesses
  
- **Regularization Techniques**:
  - L2 regularization: `λ * Σw²` added to cost function
  - Dropout: Randomly zero out a fraction of neurons during training
  - Early stopping: Stop training when validation error starts increasing
  
- **Ensemble Methods**:
  - Train multiple models with different initializations
  - Train on different subsets of data
  - Use different architectures
  - Combine predictions (voting, averaging, stacking)

### 4.2 Project Decision Framework

#### Prioritization Guidelines
- **Quick Iteration Cycle**:
  1. Start with simple model to establish baseline
  2. Perform error analysis on initial results
  3. Prioritize improvements based on error analysis
  4. Implement changes and measure impact
  5. Repeat
  
- **Orthogonalization Principle**:
  - Separate concerns for different aspects of system
  - One knob per function (avoid controls that affect multiple outcomes)
  - Example: Learning rate affects optimization, not regularization
  
- **Ablation Studies**:
  - Systematically remove components to measure their importance
  - Visualize contribution of each feature or model component
  - Quantify trade-offs between complexity and performance

## 5. Concepts Particularly Relevant for Visualization Companies

### 5.1 Data Representation

#### Tensor Visualization
- **Multi-dimensional Data Representation**:
  - Slicing techniques for high-dimensional tensors
  - Color mapping for tensor values
  - Animation for temporal dimensions
  
- **Transformation Visualization**:
  - Input → Layer 1 → Layer 2 → ... → Output
  - Visualize how data representations change through network
  - Highlight feature extraction and abstraction process

#### Feature Importance
- **Attribution Methods**:
  - Gradient-based: Saliency maps, Grad-CAM
  - Perturbation-based: Occlusion sensitivity
  - SHAP (SHapley Additive exPlanations) values
  
- **Feature Contribution**:
  - Visualize which features contribute most to predictions
  - Compare feature importance across different classes
  - Track feature importance changes during training

### 5.2 Interactive Debugging

#### Real-time Error Analysis
- **Interactive Exploration**:
  - Filter misclassifications by type, confidence, features
  - Compare similar examples with different predictions
  - Group errors by root causes
  
- **Pattern Detection**:
  - Cluster similar errors
  - Highlight systematic failure patterns
  - Provide visual cues for potential fixes

#### Hyperparameter Tuning Visualization
- **Parameter Space Exploration**:
  - Parallel coordinates plots for multi-dimensional parameters
  - Response surface visualization for parameter interactions
  - Historical tracking of parameter changes and effects
  
- **Search Strategy Visualization**:
  - Grid search results as heat maps
  - Random search distribution visualization
  - Bayesian optimization acquisition function visualization
