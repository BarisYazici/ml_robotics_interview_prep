# Machine Learning Interview Book Summary

This document summarizes the content from Chip Huyen's Machine Learning Interview Book, covering key topics and questions that ML practitioners should know for interviews.
You can support Chip Huyen from this [link](https://www.amazon.de/-/en/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969)

## Table of Contents
1. [Mathematics Foundations](#mathematics-foundations)
   - [Calculus and Convex Optimization](#calculus-and-convex-optimization)
   - [Probability and Statistics](#probability-and-statistics)
2. [Computer Science Fundamentals](#computer-science-fundamentals)
   - [Algorithms](#algorithms)
   - [Complexity and Numerical Analysis](#complexity-and-numerical-analysis)
   - [Data Structures and Data Handling](#data-structures-and-data-handling)
3. [Machine Learning Fundamentals](#machine-learning-fundamentals)
   - [ML Basics](#ml-basics)
   - [Sampling and Creating Training Data](#sampling-and-creating-training-data)
   - [Objective Functions, Metrics, and Evaluation](#objective-functions-metrics-and-evaluation)
4. [Machine Learning Methods](#machine-learning-methods)
   - [Classical Machine Learning](#classical-machine-learning)
   - [Deep Learning Architectures](#deep-learning-architectures)
      - [Natural Language Processing](#natural-language-processing)
      - [Computer Vision](#computer-vision)
      - [Reinforcement Learning](#reinforcement-learning)
   - [Training Neural Networks](#training-neural-networks)
5. [Additional Resources](#additional-resources)
   - [For Interviewers](#for-interviewers)
   - [Building Your Network](#building-your-network)

## Mathematics Foundations

### Calculus and Convex Optimization

**Overview:**
Calculus and convex optimization are fundamental to understanding how machine learning algorithms work, especially for gradient-based methods. Key topics include differentiable functions, convexity, and optimization techniques.

**Key Questions:**
1. Differentiable functions
   - What does it mean when a function is differentiable?
   - Give an example of when a function doesn't have a derivative at a point
   - Give an example of non-differentiable functions used in ML and how backpropagation works with them

2. Convexity
   - What makes a function convex or concave?
   - Why is convexity desirable in optimization?
   - Show that cross-entropy loss is convex

3. Gradient-based methods
   - First-order vs. second-order optimization methods
   - How to use the Hessian matrix to test for critical points
   - Pros and cons of second-order optimization

4. Jensen's inequality and chain rule
   - Applications in ML algorithms
   - Derivative calculations for ML functions
   - Constrained optimization techniques

**Important Notes:**
- Convex optimization is well-understood, and analyzing non-convex functions as if they were convex can provide meaningful bounds
- Stephen Boyd's textbook is recommended for convex optimization
- The Hessian matrix is used for large-scale optimization problems and expressing image processing operators

### Probability and Statistics

**Overview:**
Probability and statistics form the foundation of machine learning. Understanding concepts like distributions, confidence intervals, statistical significance, and correlation is crucial for building effective models.

**Key Questions:**
1. Probability basics
   - Uniform random variables and probability distributions
   - PDF interpretation and values
   - Independence of variables
   - Transforming distributions

2. Statistical concepts
   - Frequentist vs Bayesian statistics
   - Mean, median, variance calculations
   - Moments of functions
   - Confidence intervals interpretation

3. Practical applications
   - Handling rare events and class imbalance
   - Conditional probability calculations
   - Correlation interpretation and limitations
   - Maximum likelihood estimation

4. Statistical significance
   - Assessing meaningful patterns vs. chance
   - P-values and their distribution
   - A/B testing methods and considerations
   - Correlation between variables

**Important Notes:**
- "Data science is just doing statistics on a Mac"
- Knowledge of probability and statistics is essential for understanding objective functions in ML
- Subtle interpretation of confidence intervals is often misunderstood

## Computer Science Fundamentals

### Algorithms

**Overview:**
Knowledge of classical algorithms and programming techniques is essential for implementing efficient machine learning solutions.

**Key Problems:**
1. Recursive JSON file reading
2. O(N log N) sorting algorithms
3. Sequence problems (longest increasing subsequence, longest common subsequence)
4. Tree traversals (pre-order, in-order, post-order)
5. Continuous subarray sum problems
6. Finding median of sorted arrays
7. Matrix manipulation and calculations
8. Dynamic memory allocation
9. Mathematical expression parsing and calculation
10. File content processing and duplicate detection

### Complexity and Numerical Analysis

**Overview:**
Understanding computational complexity and numerical stability is crucial for scaling ML models and ensuring reliable performance.

**Key Questions:**
1. Matrix multiplication optimization
2. Numerical instability causes in deep learning
3. Purpose of epsilon term in algorithms
4. GPU advantages for deep learning vs TPUs
5. Problem intractability
6. Time and space complexity for backpropagation in RNNs
7. Model scaling across multiple GPUs
8. Precision reduction benefits and challenges
9. Batch normalization implementation across GPUs
10. Code optimization for vectorized operations

### Data Structures and Data Handling

**Overview:**
Effective data management is critical for ML applications, from storing and processing to monitoring data pipelines.

**Key Data Structures:**
- Trees (binary search trees, heaps, tries)
- Queues, stacks, priority queues
- Linked lists
- HashMaps and HashTables

**Data Format Considerations:**
- Row-based formats (CSV, JSON) vs. column-based formats (Parquet, ORC)
- Tradeoffs between write and read efficiency
- Framework choices for data manipulation (pandas, dask)
- Visualization libraries (seaborn, matplotlib, Tableau, ggplot)
- Big data systems (Spark, Hadoop)
- Database query languages (SQL)

## Machine Learning Fundamentals

### ML Basics

**Overview:**
Core concepts of machine learning that underpin various approaches and methodologies.

**Key Questions:**
1. Learning paradigms
   - Supervised, unsupervised, weakly supervised, semi-supervised, and active learning
   - Empirical risk minimization concepts

2. Model selection principles
   - Occam's razor in ML
   - Universal Approximation Theorem limitations
   - Hyperparameter tuning importance

3. Model categories
   - Classification vs. regression
   - Parametric vs. non-parametric methods
   - Model ensembling benefits

4. Regularization
   - L1 vs. L2 regularization effects
   - Performance degradation in production

5. Troubleshooting
   - Identifying causes for poor production performance
   - Validation techniques for hypotheses
   - Solutions for production issues

### Sampling and Creating Training Data

**Overview:**
Effective data sampling and preparation strategies are crucial for model training and performance.

**Key Questions:**
1. Sampling methods
   - With vs. without replacement
   - MCMC sampling
   - High-dimensional data sampling
   - Candidate sampling for classification

2. Practical sampling challenges
   - Selecting data for labeling
   - Evaluating label quality
   - Handling selection bias
   - Distribution testing between sets

3. Data quality issues
   - Outlier detection and handling
   - Sample duplication considerations
   - Missing data strategies
   - Class imbalance approaches

4. Data leakage
   - Training data leakage prevention
   - Feature leakage causes and detection
   - Time-based data splitting considerations

### Objective Functions, Metrics, and Evaluation

**Overview:**
Understanding how to properly evaluate models and select appropriate metrics is essential for ML success.

**Key Questions:**
1. Convergence and fitting
   - Algorithm convergence definition
   - Loss curves for overfitting and underfitting
   - Bias-variance tradeoff

2. Validation methods
   - Cross-validation techniques
   - Train/valid/test splits purpose
   - Handling problematic loss curves

3. Classification metrics
   - F1 score benefits over accuracy
   - Confusion matrix interpretation
   - Handling class imbalance in metrics

4. Regression metrics
   - Log loss vs. MSE for logistic regression
   - RMSE vs. MAE selection criteria
   - Entropy and probability distribution metrics

## Machine Learning Methods

### Classical Machine Learning

**Overview:**
Classical machine learning methods remain powerful tools for many problems and form the foundation for understanding more complex approaches.

**Key Questions:**
1. Linear models
   - Linear regression assumptions
   - Feature scaling for logistic regression
   - Fraud detection algorithms

2. Feature selection
   - Purpose and algorithms
   - Pros and cons of different methods

3. Clustering
   - K-means parameter selection
   - Evaluation with and without labels
   - K-means vs. GMM comparison

4. Ensemble methods
   - Bagging vs. boosting differences
   - Random forest vs. XGBoost
   - Application in deep learning

5. Collaborative filtering
   - User-item vs. item-item matrices
   - Handling new users
   - Naive Bayes classification
   - Support Vector Machines

### Deep Learning Architectures

#### Natural Language Processing

**Overview:**
NLP techniques for processing and understanding human language data.

**Key Questions:**
1. Recurrent architectures
   - RNN motivation
   - LSTM benefits
   - Dropout in RNNs

2. Language modeling
   - Density estimation
   - Supervised vs. unsupervised paradigms
   - N-gram vs. neural models

3. Word representations
   - Word embedding necessity
   - Count-based vs. prediction-based embeddings
   - Context-based embedding limitations

4. Model evaluation
   - TF/IDF ranking
   - Levenshtein distance
   - BLEU metric pros and cons
   - Entropy evaluation

#### Computer Vision

**Overview:**
Computer vision techniques for processing and understanding visual data.

**Key Questions:**
1. Convolutional networks
   - Filter visualization methods
   - Filter size impacts
   - "Locally connected" meaning

2. Architectural elements
   - Zero padding purpose
   - Upsampling techniques
   - 1x1 convolutional layers
   - Pooling variants and effects

3. Optimizations
   - Depthwise separable convolutions
   - Transfer learning with different image sizes
   - Converting fully-connected to convolutional layers
   - FFT-based vs. Winograd-based convolution

#### Reinforcement Learning

**Overview:**
Reinforcement learning approaches for decision-making systems.

**Key Questions:**
1. Fundamental concepts
   - Explore vs. exploit tradeoff
   - Finite vs. infinite horizons
   - Discount term purpose
   - Minimax algorithm

2. Policy approaches
   - Deriving reward functions
   - On-policy vs. off-policy
   - Model-based vs. model-free methods

### Training Neural Networks

**Overview:**
Techniques and considerations for effectively training neural network models.

**Key Questions:**
1. Training approaches
   - Overfit vs. underfit priority
   - Gradient update implementation
   - Neural network implementation in NumPy

2. Activation functions
   - Sigmoid, tanh, ReLU, leaky ReLU properties
   - Pros and cons of each
   - ReLU differentiability handling

3. Architecture considerations
   - Skip connection motivation
   - Vanishing and exploding gradients
   - Weight normalization benefits

4. Optimization techniques
   - Early stopping criteria
   - Gradient descent variants
   - Epoch-based training benefits
   - Weight fluctuatio n handling

5. Hyperparameters
   - Learning rate selection and warmup
   - Batch normalization vs. layer norm
   - Weight decay purpose
   - Batch size effects

6. Optimizer selection
   - Adagrad for sparse gradients
   - Adam vs. SGD comparison
   - Synchronous vs. asynchronous SGD

7. Model development practices
   - Weight initialization
   - Stochasticity sources and benefits
   - Dead neuron prevention
   - Pruning techniques
   - Knowledge distillation advantages

## Additional Resources

### For Interviewers

**Overview:**
The hiring process for ML roles presents unique challenges and opportunities for improvement.

**Key Points:**
- ML interviews remain a "black box" with little quantitative research on effectiveness
- Testing the interview pipeline on existing employees can provide valuable feedback
- Interviewer training programs are essential for standardization
- Candidates' experience during interviews influences their perception of the company

### Building Your Network

**Overview:**
Networking is essential for career development in the ML field.

**Key Strategies:**
- Leverage connections from school and work
- Find people with shared interests rather than forcing connections
- Engage with others' work and publish your own
- Utilize social networks like Twitter where ML professionals are active
- Attend conferences and workshops
- Build relationships naturally through shared interests
- Create an impressive portfolio to attract opportunities
- Reach out to your network when job searching

