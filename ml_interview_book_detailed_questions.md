# Machine Learning Interview Book - Detailed Questions

This document contains a comprehensive list of all questions from Chip Huyen's Machine Learning Interview Book, organized by topic.
You can support Chip Huyen from this [link](https://www.amazon.de/-/en/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969)

*Note: This summary contains only the questions from the book chapters. The answers to these questions are not provided in the available materials.*

## Mathematics Foundations

### Calculus and Convex Optimization

1. Differentiable functions
   - [E] What does it mean when a function is differentiable?
   - [E] Give an example of when a function doesn't have a derivative at a point.
   - [M] Give an example of non-differentiable functions that are frequently used in machine learning. How do we do backpropagation if those functions aren't differentiable?

2. Convexity
   - [E] What does it mean for a function to be convex or concave? Draw it.
   - [E] Why is convexity desirable in an optimization problem?
   - [M] Show that the cross-entropy loss function is convex.

3. Logistic discriminant classifier questions:
   - Show that p(y=-1|x) = σ(-w^Tx).
   - Show that ∆_wL(y_i, x_i; w) = -y_i(1-p(y_i|x_i))x_i.
   - Show that ∆_wL(y_i, x_i; w) is convex.

4. Derivatives questions:
   - [E] How can we use second-order derivatives for training models?
   - [M] Pros and cons of second-order optimization.
   - [M] Why don't we see more second-order optimization in practice?

5. [M] How can we use the Hessian (second derivative matrix) to test for critical points?
6. [E] Jensen's inequality forms the basis for many algorithms for probabilistic inference. Explain what Jensen's inequality is.
7. [E] Explain the chain rule.
8. [M] Let x ∈ R_n, L = crossentropy(softmax(x), y) in which y is a one-hot vector. Take the derivative of L with respect to x.
9. [M] Given the function f(x, y) = 4x^2 - y with the constraint x^2 + y^2 = 1. Find the function's maximum and minimum values.

### Probability and Statistics

#### Probability Questions

1. [E] Given a uniform random variable X in the range of [0, 1] inclusively. What's the probability that X = 0.5?
2. [E] Can the values of PDF be greater than 1? If so, how do we interpret PDF?
3. [E] What's the difference between multivariate distribution and multimodal distribution?
4. [E] What does it mean for two variables to be independent?
5. [E] It's a common practice to assume an unknown variable to be of the normal distribution. Why is that?
6. [E] How would you turn a probabilistic model into a deterministic model?
7. [H] Is it possible to transform non-normal variables into normal variables? How?
8. [M] When is the t-distribution useful?
9. File storage system crash probability:
   - [M] What's the probability that it will crash in the next month?
   - [M] What's the probability that it will crash at any given moment?
10. [M] Say you built a classifier to predict the outcome of football matches. In the past, it's made 10 wrong predictions out of 100. Assume all predictions are made independently, what's the probability that the next 20 predictions are all correct?
11. [M] Given two random variables X and Y. We have the values P(X|Y) and P(Y) for all values of X and Y. How would you calculate P(X)?
12. [M] You know that your colleague Jason has two children and one of them is a boy. What's the probability that Jason has two sons?
13. Electronic chip defects:
    - [E] If you randomly pick a chip from the store, what is the probability that it is defective? (When there are only two manufacturers with different defect rates)
    - [M] Suppose you now get two chips coming from the same company, but you don't know which one. When you test the first chip, it appears to be functioning. What is the probability that the second electronic chip is also good?
14. Rare disease diagnosis:
    - [E] Given a person is diagnosed positive, what's the probability that this person actually has the disease?
    - [M] What's the probability that a person has the disease if two independent tests both come back positive?
15. [M] A dating site allows users to select 10 out of 50 adjectives to describe themselves. Two users are said to match if they share at least 5 adjectives. If Jack and Jin randomly pick adjectives, what is the probability that they match?
16. [M] Consider a person A whose sex we don't know. We know that for the general human height, there are two distributions: the height of males follows h_m = N(μ_m, σ_m^2) and the height of females follows h_j = N(μ_j, σ_j^2). Derive a probability density function to describe A's height.
17. [H] There are three weather apps, each the probability of being wrong ⅓ of the time. What's the probability that it will be foggy in San Francisco tomorrow if all the apps predict that it's going to be foggy in San Francisco tomorrow and during this time of the year, San Francisco is foggy 50% of the time?
18. [M] Given n samples from a uniform distribution [0, d]. How do you estimate d? (German tank problem)
19. [M] You're drawing from a random variable that is normally distributed, X ~ N(0,1), once per day. What is the expected number of days that it takes to draw a value that's higher than 0.5?
20. [M] You're part of a class. How big the class has to be for the probability of at least a person sharing the same birthday with you is greater than 50%?
21. [H] You decide to fly to Vegas for a weekend. You pick a table that doesn't have a bet limit, and for each game, you have the probability p of winning, which doubles your bet, and 1-p of losing your bet. Assume that you have unlimited money, is there a betting strategy that has a guaranteed positive payout, regardless of the value of p?
22. [H] Given a fair coin, what's the number of flips you have to do to get two consecutive heads?
23. [H] In national health research in the US, the results show that the top 3 cities with the lowest rate of kidney failure are cities with populations under 5,000. Doctors originally thought that there must be something special about small town diets, but when they looked at the top 3 cities with the highest rate of kidney failure, they are also very small cities. What might be a probabilistic explanation for this phenomenon?
24. [M] Derive the maximum likelihood estimator of an exponential distribution.

#### Statistics Questions

1. [E] Explain frequentist vs. Bayesian statistics.
2. [E] Given the array [1, 5, 3, 2, 4, 4], find its mean, median, variance, and standard deviation.
3. [M] When should we use median instead of mean? When should we use mean instead of median?
4. [M] What is a moment of function? Explain the meanings of the zeroth to fourth moments.
5. [M] Are independence and zero covariance the same? Give a counterexample if not.
6. [E] Suppose that you take 100 random newborn puppies and determine that the average weight is 1 pound with the population standard deviation of 0.12 pounds. Assuming the weight of newborn puppies follows a normal distribution, calculate the 95% confidence interval for the average weight of all newborn puppies.
7. [M] Suppose that we examine 100 newborn puppies and the 95% confidence interval for their average weight is [0.9, 1.1] pounds. Which of the following statements is true?
   - Given a random newborn puppy, its weight has a 95% chance of being between 0.9 and 1.1 pounds.
   - If we examine another 100 newborn puppies, their mean has a 95% chance of being in that interval.
   - We're 95% confident that this interval captured the true mean weight.
8. [H] Suppose we have a random variable X supported on [0, 1] from which we can draw samples. How can we come up with an unbiased estimate of the median of X?
9. [H] Can correlation be greater than 1? Why or why not? How to interpret a correlation value of 0.3?
10. Newborn puppy weight statistics:
    - [E] Calculate your puppy's z-score if it weighs 1.1 pounds (given mean = 1, std = 0.12)
    - [E] How much does your newborn puppy have to weigh to be in the top 10% in terms of weight?
    - [M] Suppose the weight of newborn puppies followed a skew distribution. Would it still make sense to calculate z-scores?
11. [H] Tossing a coin ten times resulted in 10 heads and 5 tails. How would you analyze whether a coin is fair?
12. Statistical significance:
    - [E] How do you assess the statistical significance of a pattern whether it is a meaningful pattern or just by chance?
    - [E] What's the distribution of p-values?
    - [H] Recently, a lot of scientists started a war against statistical significance. What do we need to keep in mind when using p-value and statistical significance?
13. Variable correlation:
    - [M] What happens to a regression model if two of their supposedly independent variables are strongly correlated?
    - [M] How do we test for independence between two categorical variables?
    - [H] How do we test for independence between two continuous variables?
14. [E] A/B testing is a method of comparing two versions of a solution against each other to determine which one performs better. What are some of the pros and cons of A/B testing?
15. [M] You want to test which of the two ad placements on your website is better. How many visitors and/or how many times each ad is clicked do we need so that we can be 95% sure that one placement is better?
16. [M] Your company runs a social network whose revenue comes from showing ads in newsfeed. To double revenue, your coworker suggests that you should just double the number of ads shown. Is that a good idea? How do you find out?
17. Stock correlation pattern analysis:
    - [E] After calculating correlations of pairs of stock, you found a pair with correlation above 0.8. What's the probability that this happens by chance?
    - [M] How to avoid this kind of accidental patterns?
18. [H] How are sufficient statistics and Information Bottleneck Principle used in machine learning?m

## Computer Science Fundamentals

### Algorithms

1. Write a Python function to recursively read a JSON file.
2. Implement an O(N log N) sorting algorithm, preferably quick sort or merge sort.
3. Find the longest increasing subsequence in a string.
4. Find the longest common subsequence between two strings.
5. Traverse a tree in pre-order, in-order, and post-order.
6. Given an array of integers and an integer k, find the total number of continuous subarrays whose sum equals k. The solution should have O(N) runtime.
7. There are two sorted arrays nums1 and nums2 with m and n elements respectively. Find the median of the two sorted arrays. The solution should have O(log(m+n)) runtime.
8. Write a program to solve a Sudoku puzzle by filling the empty cells.
9. Given a memory block represented by an empty array, write a program to manage the dynamic allocation of that memory block. The program should support two methods: `malloc()` to allocate memory and `free()` to free a memory block.
10. Given a string of mathematical expression, such as `10 * 4 + (4 + 3) / (2 - 1)`, calculate it. It should support four operators `+`, `-`, `:`, `/`, and the brackets `()`.
11. Given a directory path, descend into that directory and find all the files with duplicated content.
12. In Google Docs, you have the `Justify alignment` option that spaces your text to align with both left and right margins. Write a function to print out a given text line-by-line (except the last line) in Justify alignment format. The length of a line should be configurable.
13. You have 1 million text files, each is a news article scraped from various news sites. Since news sites often report the same news, even the same articles, many of the files have content very similar to each other. Write a program to filter out these files so that the end result contains only files that are sufficiently different from each other in the language of your choice. You're free to choose a metric to define the "similarity" of content between files.

### Complexity and Numerical Analysis

1. Matrix multiplication:
   - [E] You have three matrices: A ∈ R^(100×5), B ∈ R^(5×200), C ∈ R^(200×20) and you need to calculate the product ABC. In what order would you perform your multiplication and why?
   - [M] Now you need to calculate the product of N matrices A_1A_2...A_n. How would you determine the order in which to perform the multiplication?
2. [E] What are some of the causes for numerical instability in deep learning?
3. [E] In many machine learning techniques (e.g. batch norm), we often see a small term ε added to the calculation. What's the purpose of that term?
4. [E] What made GPUs popular for deep learning? How are they compared to TPUs?
5. [M] What does it mean when we say a problem is intractable?
6. [H] What are the time and space complexity for doing backpropagation on a recurrent neural network?
7. [H] Is knowing a model's architecture and its hyperparameters enough to calculate the memory requirements for that model?
8. [H] Your model works fine on a single GPU but gives poor results when you train it on 8 GPUs. What might be the cause of this? What would you do to address it?
9. [H] What benefits do we get from reducing the precision of our model? What problems might we run into? How to solve these problems?
10. [H] How to calculate the average of 1M floating-point numbers with minimal loss of precision?
11. [H] How should we implement batch normalization if a batch is spread out over multiple GPUs?
12. [M] Given a code snippet for calculating points within a radius using nested loops, what might be a problem with it and how would you improve it?

### Data Structures and Data Handling

Key data structures to know:
- Trees: binary search tree, heap, trie (prefix and suffix tree)
- Queues, stacks, priority queues
- Linked lists
- HashMap and HashTable

Data format considerations:
- Row-based formats (CSV, JSON) vs. column-based formats (Parquet, ORC)
- Write efficiency vs. read efficiency
- Data manipulation frameworks (pandas, dask)
- Visualization libraries (seaborn, matplotlib, Tableau, ggplot)
- Distributed data systems (Spark, Hadoop)
- SQL for database queries

## Machine Learning Fundamentals

### ML Basics

1. [E] Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.
2. Empirical risk minimization:
   - [E] What's the risk in empirical risk minimization?
   - [E] Why is it empirical?
   - [E] How do we minimize that risk?
3. [E] Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?
4. [E] What are the conditions that allowed deep learning to gain popularity in the last decade?
5. [M] If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?
6. [H] The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can't a simple neural network reach an arbitrarily small positive error?
7. [E] What are saddle points and local minima? Which are thought to cause more problems for training large NNs?
8. Hyperparameters:
   - [E] What are the differences between parameters and hyperparameters?
   - [E] Why is hyperparameter tuning important?
   - [M] Explain algorithm for tuning hyperparameters.
9. Classification vs. regression:
   - [E] What makes a classification problem different from a regression problem?
   - [E] Can a classification problem be turned into a regression problem and vice versa?
10. Parametric vs. non-parametric methods:
    - [E] What's the difference between parametric methods and non-parametric methods? Give an example of each method.
    - [H] When should we use one and when should we use the other?
11. [M] Why does ensembling independently trained models generally improve performance?
12. [M] Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?
13. [E] Why does an ML model's performance degrade in production?
14. [M] What problems might we run into when deploying large machine learning models?
15. Your model performs really well on the test set but poorly in production:
    - [M] What are your hypotheses about the causes?
    - [H] How do you validate whether your hypotheses are correct?
    - [M] Imagine your hypotheses about the causes are correct. What would you do to address them?

### Sampling and Creating Training Data

1. [E] If you have 6 shirts and 4 pairs of pants, how many ways are there to choose 2 shirts and 1 pair of pants?
2. [M] What is the difference between sampling with vs. without replacement? Name an example of when you would use one rather than the other?
3. [M] Explain Markov chain Monte Carlo sampling.
4. [M] If you need to sample from high-dimensional data, which sampling method would you choose?
5. [H] Suppose we have a classification task with many classes. An example is when you have to predict the next word in a sentence -- the next word can be one of many, many possible words. If we have to calculate the probabilities for all classes, it'll be prohibitively expensive. Instead, we can calculate the probabilities for a small set of candidate classes. This method is called candidate sampling. Name and explain some of the candidate sampling algorithms.
6. Reddit comment classification sampling:
   - [M] How would you sample 100K comments to label from 10 million unlabeled comments?
   - [M] Suppose you get back 100K labeled comments from 20 annotators and you want to look at some labels to estimate the quality of the labels. How many labels would you look at? How would you sample them?
7. [M] Suppose you work for a news site that historically has translated only 1% of all its articles. Your coworker argues that we should translate more articles into Chinese because translations help with the readership. On average, your translated articles have twice as many views as your non-translated articles. What might be wrong with this argument?
8. [M] How to determine whether two sets of samples (e.g. train and test splits) come from the same distribution?
9. [H] How do you know you've collected enough samples to train your ML model?
10. [M] How to determine outliers in your data samples? What to do with them?
11. Sample duplication:
    - [M] When should you remove duplicate training samples? When shouldn't you?
    - [M] What happens if we accidentally duplicate every data point in your train set or in your test set?
12. Missing data:
    - [H] In your dataset, two out of 20 variables have more than 30% missing values. What would you do?
    - [M] How might techniques that handle missing data make selection bias worse? How do you handle this bias?
13. [M] Why is randomization important when designing experiments (experimental design)?
14. Class imbalance:
    - [E] How would class imbalance affect your model?
    - [E] Why is it hard for ML models to perform well on data with class imbalance?
    - [M] Imagine you want to build a model to detect skin legions from images. In your training dataset, only 1% of your images shows signs of legions. After training, your model seems to make a lot more false negatives than false positives. What are some of the techniques you'd use to improve your model?
15. Training data leakage:
    - [M] Imagine you're working with a binary task where the positive class accounts for only 1% of your data. You decide to oversample the rare class then split your data into train and test splits. Your model performs well on the test split but poorly in production. What might have happened?
    - [M] You want to build a model to classify whether a comment is spam or not spam. You have a dataset of a million comments over the period of 7 days. You decide to randomly split all your data into the train and test splits. Your co-worker points out that this can lead to data leakage. How?
16. [M] How does data sparsity affect your models?
17. Feature leakage:
    - [E] What are some causes of feature leakage?
    - [E] Why does normalization help prevent feature leakage?
    - [M] How do you detect feature leakage?
18. [M] Suppose you want to build a model to classify whether a tweet spreads misinformation. You have 100K labeled tweets over the last 24 months. You decide to randomly shuffle on your data and pick 80% to be the train split, 10% to be the valid split, and 10% to be the test split. What might be the problem with this way of partitioning?
19. [M] You're building a neural network and you want to use both numerical and textual features. How would you process those different features?
20. [H] Your model has been performing fairly well using just a subset of features available in your data. Your boss decided that you should use all the features available instead. What might happen to the training error? What might happen to the test error?

### Objective Functions, Metrics, and Evaluation

1. Convergence:
   - [E] When we say an algorithm converges, what does convergence mean?
   - [E] How do we know when a model has converged?
2. [E] Draw the loss curves for overfitting and underfitting.
3. Bias-variance trade-off:
   - [E] What's the bias-variance trade-off?
   - [M] How's this tradeoff related to overfitting and underfitting?
   - [M] How do you know that your model is high variance, low bias? What would you do in this case?
   - [M] How do you know that your model is low variance, high bias? What would you do in this case?
4. Cross-validation:
   - [E] Explain different methods for cross-validation.
   - [M] Why don't we see more cross-validation in deep learning?
5. Train, valid, test splits:
   - [E] What's wrong with training and testing a model on the same data?
   - [E] Why do we need a validation set on top of a train set and a test set?
   - [M] How would you respond if your model's loss curves on train, valid, and test sets show problematic patterns?
6. [E] Your team is building a system to aid doctors in predicting whether a patient has cancer or not from their X-ray scan. Your colleague announces that the problem is solved now that they've built a system that can predict with 99.99% accuracy. How would you respond to that claim?
7. F1 score:
   - [E] What's the benefit of F1 over the accuracy?
   - [M] Can we still use F1 for a problem with more than two classes. How?
8. Given a binary classifier's confusion matrix:
   - [E] Calculate the model's precision, recall, and F1.
   - [M] What can we do to improve the model's performance?
9. Imbalanced classification (99% class A, 1% class B):
   - [M] If your model predicts A 100% of the time, what would the F1 score be?
   - [M] If we have a model that predicts A and B at random, what would the expected F1 be?
10. [M] For logistic regression, why is log loss recommended over MSE (mean squared error)?
11. [M] When should we use RMSE (Root Mean Squared Error) over MAE (Mean Absolute Error) and vice versa?
12. [M] Show that the negative log-likelihood and cross-entropy are the same for binary classification tasks.
13. [M] For classification tasks with more than two labels (e.g. MNIST with 10 labels), why is cross-entropy a better loss function than MSE?
14. [E] Consider a language with an alphabet of 27 characters. What would be the maximal entropy of this language?
15. [E] A lot of machine learning models aim to approximate probability distributions. Let's say P is the distribution of the data and Q is the distribution learned by our model. How do measure how close Q is to P?
16. MPE (Most Probable Explanation) vs. MAP (Maximum A Posteriori):
    - [E] How do MPE and MAP differ?
    - [H] Give an example of when they would produce different results.
17. [E] Suppose you want to build a model to predict the price of a stock in the next 8 hours and that the predicted price should never be off more than 10% from the actual price. Which metric would you use?

## Machine Learning Methods

### Classical Machine Learning

1. [E] What are the basic assumptions to be made for linear regression?
2. [E] What happens if we don't apply feature scaling to logistic regression?
3. [E] What are the algorithms you'd use when developing the prototype of a fraud detection model?
4. Feature selection:
   - [E] Why do we use feature selection?
   - [M] What are some of the algorithms for feature selection? Pros and cons of each.
5. k-means clustering:
   - [E] How would you choose the value of k?
   - [E] If the labels are known, how would you evaluate the performance of your k-means clustering algorithm?
   - [M] How would you evaluate it if the labels aren't known?
   - [H] Given a specific dataset, can you predict how K-means clustering works on it?
6. k-nearest neighbor classification:
   - [E] How would you choose the value of k?
   - [E] What happens when you increase or decrease the value of k?
   - [M] How does the value of k impact the bias and variance?
7. k-means and GMM comparison:
   - [M] Compare the two clustering algorithms.
   - [M] When would you choose one over another?
8. Bagging vs. boosting:
   - [M] What are some of the fundamental differences between bagging and boosting algorithms?
   - [M] How are they used in deep learning?
9. Graph analysis:
   - [E] Construct the adjacency matrix for a given directed graph.
   - [E] How would this matrix change if the graph is now undirected?
   - [M] What can you say about the adjacency matrices of two isomorphic graphs?
10. Collaborative filtering:
    - [M] You can build either a user-item matrix or an item-item matrix. What are the pros and cons of each approach?
    - [E] How would you handle a new user who hasn't made any purchases in the past?
11. [E] Is feature scaling necessary for kernel methods?
12. Naive Bayes:
    - [E] How is Naive Bayes classifier naive?
    - [M] Construct a Naive Bayes classifier for tweet sentiment analysis using given examples.
13. Gradient boosting:
    - [E] What is gradient boosting?
    - [M] What problems is gradient boosting good for?
14. SVM:
    - [E] What's linear separation? Why is it desirable when we use SVM?
    - [M] How well would vanilla SVM work on various example datasets?

### Deep Learning Architectures

#### Natural Language Processing

1. RNNs:
   - [E] What's the motivation for RNN?
   - [E] What's the motivation for LSTM?
   - [M] How would you do dropouts in an RNN?
2. [E] What's density estimation? Why do we say a language model is a density estimator?
3. [M] Language models are often referred to as unsupervised learning, but some say its mechanism isn't that different from supervised learning. What are your thoughts?
4. Word embeddings:
   - [M] Why do we need word embeddings?
     Answer: Word embeddings are essential because they:
     - Convert discrete word tokens into continuous vector spaces where semantic relationships are preserved
     - Reduce dimensionality compared to one-hot encoding (e.g., from vocabulary size to 300 dimensions)
     - Enable meaningful mathematical operations on words (like finding analogies)
     - Allow neural networks to process words effectively
     - Help models generalize across similar words (e.g., "apple" and "orange" will have similar representations)
     - Enable transfer learning from large text corpora to smaller tasks

   - [M] What's the difference between count-based and prediction-based word embeddings?
     Answer:
     Count-based embeddings:
     - Based on co-occurrence statistics of words in corpus
     - Examples include LSA, HAL, COALS, and GloVe
     - Process involves counting co-occurrences then applying dimensionality reduction
     - Often use SVD or similar techniques for dimension reduction
     - Computationally efficient for training

     Prediction-based embeddings:
     - Learn word vectors by predicting context words
     - Examples include Word2Vec (CBOW, Skip-gram), ELMo
     - Usually trained with neural networks
     - Often capture more semantic information
     - Can be more computationally intensive to train
     - Generally perform better on semantic tasks

   - [H] Most word embedding algorithms are based on the assumption that words that appear in similar contexts have similar meanings. What are some of the problems with context-based word embeddings?
     Answer: Key problems include:
     - Polysemy: Single representation for words with multiple meanings (e.g., "bank" as financial institution vs. river bank)
     - Static nature: Each word has the same vector regardless of context
     - Window-based context may miss long-range dependencies
     - May encode societal biases present in the training corpus
     - Difficulty handling rare words or out-of-vocabulary words
     - Cannot capture compositional semantics well (meaning of phrases)
     - Newer contextual embeddings (BERT, ELMo) address some of these limitations by generating dynamic embeddings based on context

5. TF/IDF ranking:
   - [M] Given a query and a set of documents, find the top-ranked documents according to TF/IDF.
   - [M] How document ranking changes when term frequency changes within a document?
6. [E] Your client wants you to train a language model on their dataset but their dataset is very small with only about 10,000 tokens. Would you use an n-gram or a neural language model?
7. [E] For n-gram language models, does increasing the context length (n) improve the model's performance? Why or why not?
8. [M] What problems might we encounter when using softmax as the last layer for word-level language models? How do we fix it?
9. [E] What's the Levenshtein distance of the two words "doctor" and "bottle"?
10. [M] BLEU is a popular metric for machine translation. What are the pros and cons of BLEU?
11. [H] On the same test set, LM model A has a character-level entropy of 2 while LM model A has a word-level entropy of 6. Which model would you choose to deploy?
12. [M] Imagine you have to train a NER model on the text corpus A. Would you make A case-sensitive or case-insensitive?
13. [M] Why does removing stop words sometimes hurt a sentiment analysis model?
14. [M] Many models use relative position embedding instead of absolute position embedding. Why is that?
15. [H] Some NLP models use the same weights for both the embedding layer and the layer just before softmax. What's the purpose of this?

#### Computer Vision

1. [M] For neural networks that work with images like VGG-19, InceptionNet, you often see a visualization of what type of features each filter captures. How are these visualizations created?
2. Filter size:
   - [M] How are your model's accuracy and computational efficiency affected when you decrease or increase its filter size?
   - [E] How do you choose the ideal filter size?
3. [M] Convolutional layers are also known as "locally connected." Explain what it means.
4. [M] When we use CNNs for text data, what would the number of channels be for the first conv layer?
5. [E] What is the role of zero padding?
6. [E] Why do we need upsampling? How to do it?
7. [M] What does a 1x1 convolutional layer do?
8. Pooling:
   - [E] What happens when you use max-pooling instead of average pooling?
   - [E] When should we use one instead of the other?
   - [E] What happens when pooling is removed completely?
   - [M] What happens if we replace a 2 x 2 max pool layer with a conv layer of stride 2?
9. [M] When we replace a normal convolutional layer with a depthwise separable convolutional layer, the number of parameters can go down. How does this happen? Give an example to illustrate this.
10. [M] Can you use a base model trained on ImageNet (image size 256 x 256) for an object classification task on images of size 320 x 360? How?
11. [H] How can a fully-connected layer be converted to a convolutional layer?
12. [H] Pros and cons of FFT-based convolution and Winograd-based convolution.

#### Reinforcement Learning

1. [E] Explain the explore vs exploit tradeoff with examples.
2. [E] How would a finite or infinite horizon affect our algorithms?
3. [E] Why do we need the discount term for objective functions?
4. [E] Fill in the empty circles using the minimax algorithm (for a given game tree).
5. [M] Fill in the alpha and beta values as you traverse the minimax tree from left to right (for alpha-beta pruning).
6. [E] Given a policy, derive the reward function.
7. [M] Pros and cons of on-policy vs. off-policy.
8. [M] What's the difference between model-based and model-free? Which one is more data-efficient?

#### Other Deep Learning Topics

1. [M] An autoencoder is a neural network that learns to copy its input to its output. When would this be useful?
2. Self-attention:
   - [E] What's the motivation for self-attention?
   - [E] Why would you choose a self-attention architecture over RNNs or CNNs?
   - [M] Why would you need multi-headed attention instead of just one head for attention?
   - [M] How would changing the number of heads in multi-headed attention affect the model's performance?
3. Transfer learning:
   - [E] You want to build a classifier to predict sentiment in tweets but you have very little labeled data (say 1000). What do you do?
   - [M] What's gradual unfreezing? How might it help with transfer learning?
4. Bayesian methods:
   - [M] How do Bayesian methods differ from the mainstream deep learning approach?
   - [M] How are the pros and cons of Bayesian neural networks compared to the mainstream neural networks?
   - [M] Why do we say that Bayesian neural networks are natural ensembles?
5. GANs:
   - [E] What do GANs converge to?
   - [M] Why are GANs so hard to train?

### Training Neural Networks

1. [E] When building a neural network, should you overfit or underfit it first?
2. [E] Write the vanilla gradient update.
3. Neural network in simple Numpy:
   - [E] Write in plain NumPy the forward and backward pass for a two-layer feed-forward neural network with a ReLU layer in between.
   - [M] Implement vanilla dropout for the forward and backward pass in NumPy.
4. Activation functions:
   - [E] Draw the graphs for sigmoid, tanh, ReLU, and leaky ReLU.
   - [E] Pros and cons of each activation function.
   - [E] Is ReLU differentiable? What to do when it's not differentiable?
   - [M] Derive derivatives for sigmoid function σ(x) when x is a vector.
5. [E] What's the motivation for skip connection in neural works?
6. Vanishing and exploding gradients:
   - [E] How do we know that gradients are exploding? How do we prevent it?
   - [E] Why are RNNs especially susceptible to vanishing and exploding gradients?
7. [M] Weight normalization separates a weight vector's norm from its gradient. How would it help with training?
8. [M] When training a large neural network, say a language model with a billion parameters, you evaluate your model on a validation set at the end of every epoch. You realize that your validation loss is often lower than your train loss. What might be happening?
9. [E] What criteria would you use for early stopping?
10. [E] Gradient descent vs SGD vs mini-batch SGD.
11. [H] It's a common practice to train deep learning models using epochs: we sample batches from data **without** replacement. Why would we use epochs instead of just sampling data **with** replacement?
12. [M] Your model' weights fluctuate a lot during training. How does that affect your model's performance? What to do about it?
13. Learning rate:
    - [E] Draw a graph number of training epochs vs training error for when the learning rate is too high, too low, and acceptable.
    - [E] What's learning rate warmup? Why do we need it?
14. [E] Compare batch norm and layer norm.
15. [M] Why is squared L2 norm sometimes preferred to L2 norm for regularizing neural networks?
16. [E] Some models use weight decay: after each gradient update, the weights are multiplied by a factor slightly less than 1. What is this useful for?
17. Learning rate reduction:
    - [E] What's the motivation for reducing learning rate throughout training?
    - [M] What might be the exceptions to this practice?
18. Batch size:
    - [E] What happens to your model training when you decrease the batch size to 1?
    - [E] What happens when you use the entire training data in a batch?
    - [M] How should we adjust the learning rate as we increase or decrease the batch size?
19. [M] Why is Adagrad sometimes favored in problems with sparse gradients?
20. Adam vs. SGD:
    - [M] What can you say about the ability to converge and generalize of Adam vs. SGD?
    - [M] What else can you say about the difference between these two optimizers?
21. [M] With model parallelism, you might update your model weights using the gradients from each machine asynchronously or synchronously. What are the pros and cons of asynchronous SGD vs. synchronous SGD?
22. [M] Why shouldn't we have two consecutive linear layers in a neural network?
23. [M] Can a neural network with only RELU (non-linearity) act as a linear classifier?
24. [M] Design the smallest neural network that can function as an XOR gate.
25. [E] Why don't we just initialize all weights in a neural network to zero?
26. Stochasticity:
    - [M] What are some sources of randomness in a neural network?
    - [M] Sometimes stochasticity is desirable when training neural networks. Why is that?
27. Dead neuron:
    - [E] What's a dead neuron?
    - [E] How do we detect them in our neural network?
    - [M] How to prevent them?
28. Pruning:
    - [M] Pruning is a popular technique where certain weights of a neural network are set to 0. Why is it desirable?
    - [M] How do you choose what to prune from a neural network?
29. [H] Under what conditions would it be possible to recover training data from the weight checkpoints?
30. [H] Why do we try to reduce the size of a big trained model through techniques such as knowledge distillation instead of just training a small model from the beginning?

*Note: Question difficulty is marked as [E] for Easy, [M] for Medium, and [H] for Hard.* 