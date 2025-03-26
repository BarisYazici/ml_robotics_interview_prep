# Machine Learning Interview Questions

This document contains a comprehensive collection of machine learning interview questions organized by topic. Questions are marked with difficulty level:
- [E] Easy
- [M] Medium 
- [H] Hard

## Table of Contents
- [Linear Algebra and Calculus](#linear-algebra-and-calculus)
  - [Vectors](#vectors)
  - [Matrices](#matrices)
  - [Dimensionality Reduction](#dimensionality-reduction)
- [Probability and Statistics](#probability-and-statistics)
  - [Probability](#probability)
  - [Statistics](#statistics)
- [Algorithms and Computational Complexity](#algorithms-and-computational-complexity)
  - [Algorithms](#algorithms)
  - [Complexity and Numerical Analysis](#complexity-and-numerical-analysis)
- [Machine Learning Basics](#machine-learning-basics)
  - [Fundamentals](#fundamentals)
  - [Sampling and Training Data](#sampling-and-training-data)
- [Classical Machine Learning](#classical-machine-learning)
- [Deep Learning Architectures and Applications](#deep-learning-architectures-and-applications)
  - [Natural Language Processing](#natural-language-processing)
  - [Computer Vision](#computer-vision)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Other Deep Learning Topics](#other-deep-learning-topics)

## Linear Algebra and Calculus

### Vectors

1. [E] What's the geometric interpretation of the dot product of two vectors?
2. [E] Given a vector u, find vector v of unit length such that the dot product of u and v is maximum.
3. [E] Given two vectors a = [3, 2, 1] and b = [-1, 0, 1]. Calculate the outer product a^Tb?
4. [M] Give an example of how the outer product can be useful in ML.
5. [E] What does it mean for two vectors to be linearly independent?
6. [M] Given two sets of vectors A = {a_1, a_2, a_3, ..., a_n} and B = {b_1, b_2, b_3, ... , b_m}. How do you check that they share the same basis?
7. [M] Given n vectors, each of d dimensions. What is the dimension of their span?
8. [E] What's a norm? What is L_0, L_1, L_2, L_{norm}?
9. [M] How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?

### Matrices

1. [E] Why do we say that matrices are linear transformations?
2. [E] What's the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?
3. [E] What does the determinant of a matrix represent?
4. [E] What happens to the determinant of a matrix if we multiply one of its rows by a scalar t × R?
5. [M] A 4 × 4 matrix has four eigenvalues 3, 3, 2, -1. What can we say about the trace and the determinant of this matrix?
6. [M] Given the following matrix:
   ```
   [1  4  -2]
   [-1 3   2]
   [3  5  -6]
   ```
   Without explicitly using the equation for calculating determinants, what can we say about this matrix's determinant?
7. [M] What's the difference between the covariance matrix A^TA and the Gram matrix AA^T?
8. [M] Find x such that: Ax = b, given A ∈ R^{n × m} and b ∈ R^n
9. [E] When does this have a unique solution?
10. [M] Why is it when A has more columns than rows, Ax = b has multiple solutions?
11. [M] Given a matrix A with no inverse. How would you solve the equation Ax = b? What is the pseudoinverse and how to calculate it?
12. [E] What does derivative represent?
13. [M] What's the difference between derivative, gradient, and Jacobian?
14. [H] Say we have the weights w ∈ R^{d × m} and a mini-batch x of n elements, each element is of the shape 1 × d so that x ∈ R^{n × d}. We have the output y = f(x; w) = xw. What's the dimension of the Jacobian ∂y/∂x?
15. [H] Given a very large symmetric matrix A that doesn't fit in memory, say A ∈ R^{1M × 1M} and a function f that can quickly compute f(x) = Ax for x ∈ R^{1M}. Find the unit vector x so that x^TAx is minimal.

### Dimensionality Reduction

1. [E] Why do we need dimensionality reduction?
2. [E] Eigendecomposition is a common factorization technique used for dimensionality reduction. Is the eigendecomposition of a matrix always unique?
3. [M] Name some applications of eigenvalues and eigenvectors.
4. [M] We want to do PCA on a dataset of multiple features in different ranges. For example, one is in the range 0-1 and one is in the range 10 - 1000. Will PCA work on this dataset?
5. [H] Under what conditions can one apply eigendecomposition? What about SVD?
6. [H] What is the relationship between SVD and eigendecomposition?
7. [H] What's the relationship between PCA and SVD?
8. [H] How does t-SNE (T-distributed Stochastic Neighbor Embedding) work? Why do we need it?

## Probability and Statistics

### Probability

1. [E] Given a uniform random variable X in the range of [0, 1] inclusively. What's the probability that X=0.5?
2. [E] Can the values of PDF be greater than 1? If so, how do we interpret PDF?
3. [E] What's the difference between multivariate distribution and multimodal distribution?
4. [E] What does it mean for two variables to be independent?
5. [E] It's a common practice to assume an unknown variable to be of the normal distribution. Why is that?
6. [E] How would you turn a probabilistic model into a deterministic model?
7. [H] Is it possible to transform non-normal variables into normal variables? How?
8. [M] When is the t-distribution useful?
9. [M] Assume you manage an unreliable file storage system that crashed 5 times in the last year, each crash happens independently. What's the probability that it will crash in the next month?
10. [M] Assume you manage an unreliable file storage system that crashed 5 times in the last year, each crash happens independently. What's the probability that it will crash at any given moment?
11. [M] Say you built a classifier to predict the outcome of football matches. In the past, it's made 10 wrong predictions out of 100. Assume all predictions are made independently, what's the probability that the next 20 predictions are all correct?
12. [M] Given two random variables X and Y. We have the values P(X|Y) and P(Y) for all values of X and Y. How would you calculate P(X)?
13. [M] You know that your colleague Jason has two children and one of them is a boy. What's the probability that Jason has two sons?
14. [E] If you randomly pick a chip from the store where there are two manufacturers (A makes defective chips with 30% probability, B with 70% probability), what is the probability that it is defective?
15. [M] Suppose you now get two chips coming from the same company, but you don't know which one. When you test the first chip, it appears to be functioning. What is the probability that the second electronic chip is also good?
16. [E] Given a rare disease that only 1 in 10000 people get, with a test having false positive and false negative rates of 1%, what's the probability that a person with a positive diagnosis actually has the disease?
17. [M] What's the probability that a person has the disease if two independent tests both come back positive?
18. [M] A dating site allows users to select 10 out of 50 adjectives to describe themselves. Two users are said to match if they share at least 5 adjectives. If Jack and Jin randomly pick adjectives, what is the probability that they match?
19. [M] Consider a person A whose sex we don't know. We know that for the general human height, there are two distributions: the height of males follows h_m = N(μ_m, σ_m²) and the height of females follows h_j = N(μ_j, σ_j²). Derive a probability density function to describe A's height.
20. [H] There are three weather apps, each with the probability of being wrong ⅓ of the time. What's the probability that it will be foggy in San Francisco tomorrow if all the apps predict that it's going to be foggy in San Francisco tomorrow and during this time of the year, San Francisco is foggy 50% of the time?
21. [M] Given n samples from a uniform distribution [0, d]. How do you estimate d? (Also known as the German tank problem)
22. [M] You're drawing from a random variable that is normally distributed, X ~ N(0,1), once per day. What is the expected number of days that it takes to draw a value that's higher than 0.5?
23. [M] You're part of a class. How big the class has to be for the probability of at least a person sharing the same birthday with you is greater than 50%?
24. [H] You decide to fly to Vegas for a weekend. You pick a table that doesn't have a bet limit, and for each game, you have the probability p of winning, which doubles your bet, and 1-p of losing your bet. Assume that you have unlimited money, is there a betting strategy that has a guaranteed positive payout, regardless of the value of p?
25. [H] Given a fair coin, what's the number of flips you have to do to get two consecutive heads?
26. [H] In national health research in the US, the results show that the top 3 cities with the lowest rate of kidney failure are cities with populations under 5,000. Doctors originally thought that there must be something special about small town diets, but when they looked at the top 3 cities with the highest rate of kidney failure, they are also very small cities. What might be a probabilistic explanation for this phenomenon?
27. [M] Derive the maximum likelihood estimator of an exponential distribution.

### Statistics

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
10. [E] Calculate your puppy's z-score (standard score) if it weighs 1.1 pounds (given the weight of newborn puppies is roughly symmetric with a mean of 1 pound and a standard deviation of 0.12).
11. [E] How much does your newborn puppy have to weigh to be in the top 10% in terms of weight?
12. [M] Suppose the weight of newborn puppies followed a skew distribution. Would it still make sense to calculate z-scores?
13. [H] Tossing a coin ten times resulted in 10 heads and 5 tails. How would you analyze whether a coin is fair?
14. [E] How do you assess the statistical significance of a pattern whether it is a meaningful pattern or just by chance?
15. [E] What's the distribution of p-values?
16. [H] Recently, a lot of scientists started a war against statistical significance. What do we need to keep in mind when using p-value and statistical significance?
17. [M] What happens to a regression model if two of their supposedly independent variables are strongly correlated?
18. [M] How do we test for independence between two categorical variables?
19. [H] How do we test for independence between two continuous variables?
20. [E] A/B testing is a method of comparing two versions of a solution against each other to determine which one performs better. What are some of the pros and cons of A/B testing?
21. [M] You want to test which of the two ad placements on your website is better. How many visitors and/or how many times each ad is clicked do we need so that we can be 95% sure that one placement is better?
22. [M] Your company runs a social network whose revenue comes from showing ads in newsfeed. To double revenue, your coworker suggests that you should just double the number of ads shown. Is that a good idea? How do you find out?
23. [E] What's the probability that finding a correlation above 0.8 between a pair of stocks happens by chance (when examining 10,000 stocks over 24 months)?
24. [M] How to avoid finding accidental patterns in big data?
25. [H] How are sufficient statistics and Information Bottleneck Principle used in machine learning?

## Algorithms and Computational Complexity

### Algorithms

1. Write a Python function to recursively read a JSON file.
2. Implement an O(N log N) sorting algorithm, preferably quick sort or merge sort.
3. Find the longest increasing subsequence in a string.
4. Find the longest common subsequence between two strings.
5. Traverse a tree in pre-order, in-order, and post-order.
6. Given an array of integers and an integer k, find the total number of continuous subarrays whose sum equals k. The solution should have O(N) runtime.
7. There are two sorted arrays nums1 and nums2 with m and n elements respectively. Find the median of the two sorted arrays. The solution should have O(log(m+n)) runtime.
8. Write a program to solve a Sudoku puzzle by filling the empty cells. The board is of the size 9 × 9. It contains only 1-9 numbers. Empty cells are denoted with *. Each board has one unique solution.
9. Given a memory block represented by an empty array, write a program to manage the dynamic allocation of that memory block. The program should support two methods: `malloc()` to allocate memory and `free()` to free a memory block.
10. Given a string of mathematical expression, such as `10 * 4 + (4 + 3) / (2 - 1)`, calculate it. It should support four operators `+`, `-`, `:`, `/`, and the brackets `()`.
11. Given a directory path, descend into that directory and find all the files with duplicated content.
12. In Google Docs, you have the `Justify alignment` option that spaces your text to align with both left and right margins. Write a function to print out a given text line-by-line (except the last line) in Justify alignment format. The length of a line should be configurable.
13. You have 1 million text files, each is a news article scraped from various news sites. Since news sites often report the same news, even the same articles, many of the files have content very similar to each other. Write a program to filter out these files so that the end result contains only files that are sufficiently different from each other in the language of your choice. You're free to choose a metric to define the "similarity" of content between files.

### Complexity and Numerical Analysis

1. [E] You have three matrices: A ∈ R^{100 × 5}, B ∈ R^{5 × 200}, C ∈ R^{200 × 20} and you need to calculate the product ABC. In what order would you perform your multiplication and why?
2. [M] Now you need to calculate the product of N matrices A_1A_2...A_n. How would you determine the order in which to perform the multiplication?
3. [E] What are some of the causes for numerical instability in deep learning?
4. [E] In many machine learning techniques (e.g. batch norm), we often see a small term ε added to the calculation. What's the purpose of that term?
5. [E] What made GPUs popular for deep learning? How are they compared to TPUs?
6. [M] What does it mean when we say a problem is intractable?
7. [H] What are the time and space complexity for doing backpropagation on a recurrent neural network?
8. [H] Is knowing a model's architecture and its hyperparameters enough to calculate the memory requirements for that model?
9. [H] Your model works fine on a single GPU but gives poor results when you train it on 8 GPUs. What might be the cause of this? What would you do to address it?
10. [H] What benefits do we get from reducing the precision of our model? What problems might we run into? How to solve these problems?
11. [H] How to calculate the average of 1M floating-point numbers with minimal loss of precision?
12. [H] How should we implement batch normalization if a batch is spread out over multiple GPUs?
13. [M] Given the following code snippet. What might be a problem with it? How would you improve it?
```python
import numpy as np

def within_radius(a, b, radius):
    if np.linalg.norm(a - b) < radius:
        return 1
    return 0

def make_mask(volume, roi, radius):
    mask = np.zeros(volume.shape)
    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                mask[x, y, z] = within_radius((x, y, z), roi, radius)
    return mask
```

## Machine Learning Basics

### Fundamentals

1. [E] Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.
2. [E] What's the risk in empirical risk minimization?
3. [E] Why is it empirical?
4. [E] How do we minimize that risk?
5. [E] Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?
6. [E] What are the conditions that allowed deep learning to gain popularity in the last decade?
7. [M] If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?
8. [H] The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can't a simple neural network reach an arbitrarily small positive error?
9. [E] What are saddle points and local minima? Which are thought to cause more problems for training large NNs?
10. [E] What are the differences between parameters and hyperparameters?
11. [E] Why is hyperparameter tuning important?
12. [M] Explain algorithm for tuning hyperparameters.
13. [E] What makes a classification problem different from a regression problem?
14. [E] Can a classification problem be turned into a regression problem and vice versa?
15. [E] What's the difference between parametric methods and non-parametric methods? Give an example of each method.
16. [H] When should we use one and when should we use the other?
17. [M] Why does ensembling independently trained models generally improve performance?
18. [M] Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?
19. [E] Why does an ML model's performance degrade in production?
20. [M] What problems might we run into when deploying large machine learning models?
21. [M] Your model performs really well on the test set but poorly in production. What are your hypotheses about the causes?
22. [H] How do you validate whether your hypotheses are correct?
23. [M] Imagine your hypotheses about the causes are correct. What would you do to address them?

### Sampling and Training Data

1. [E] If you have 6 shirts and 4 pairs of pants, how many ways are there to choose 2 shirts and 1 pair of pants?
2. [M] What is the difference between sampling with vs. without replacement? Name an example of when you would use one rather than the other?
3. [M] Explain Markov chain Monte Carlo sampling.
4. [M] If you need to sample from high-dimensional data, which sampling method would you choose?
5. [H] Suppose we have a classification task with many classes. An example is when you have to predict the next word in a sentence -- the next word can be one of many, many possible words. If we have to calculate the probabilities for all classes, it'll be prohibitively expensive. Instead, we can calculate the probabilities for a small set of candidate classes. This method is called candidate sampling. Name and explain some of the candidate sampling algorithms.
6. [M] How would you sample 100K comments to label from 10 million unlabeled Reddit comments from 10K users over the last 24 months?
7. [M] Suppose you get back 100K labeled comments from 20 annotators and you want to look at some labels to estimate the quality of the labels. How many labels would you look at? How would you sample them?
8. [M] Suppose you work for a news site that historically has translated only 1% of all its articles. Your coworker argues that we should translate more articles into Chinese because translations help with the readership. On average, your translated articles have twice as many views as your non-translated articles. What might be wrong with this argument?
9. [M] How to determine whether two sets of samples (e.g. train and test splits) come from the same distribution?
10. [H] How do you know you've collected enough samples to train your ML model?
11. [M] How to determine outliers in your data samples? What to do with them?
12. [M] When should you remove duplicate training samples? When shouldn't you?
13. [M] What happens if we accidentally duplicate every data point in your train set or in your test set?
14. [H] In your dataset, two out of 20 variables have more than 30% missing values. What would you do?
15. [M] How might techniques that handle missing data make selection bias worse? How do you handle this bias?
16. [M] Why is randomization important when designing experiments (experimental design)?
17. [E] How would class imbalance affect your model?
18. [E] Why is it hard for ML models to perform well on data with class imbalance?
19. [M] Imagine you want to build a model to detect skin legions from images. In your training dataset, only 1% of your images shows signs of legions. After training, your model seems to make a lot more false negatives than false positives. What are some of the techniques you'd use to improve your model?
20. [M] Imagine you're working with a binary task where the positive class accounts for only 1% of your data. You decide to oversample the rare class then split your data into train and test splits. Your model performs well on the test split but poorly in production. What might have happened?
21. [M] You want to build a model to classify whether a comment is spam or not spam. You have a dataset of a million comments over the period of 7 days. You decide to randomly split all your data into the train and test splits. Your co-worker points out that this can lead to data leakage. How?
22. [M] How does data sparsity affect your models?
23. [E] What are some causes of feature leakage?
24. [E] Why does normalization help prevent feature leakage?
25. [M] How do you detect feature leakage?
26. [M] Suppose you want to build a model to classify whether a tweet spreads misinformation. You have 100K labeled tweets over the last 24 months. You decide to randomly shuffle on your data and pick 80% to be the train split, 10% to be the valid split, and 10% to be the test split. What might be the problem with this way of partitioning?
27. [M] You're building a neural network and you want to use both numerical and textual features. How would you process those different features?
28. [H] Your model has been performing fairly well using just a subset of features available in your data. Your boss decided that you should use all the features available instead. What might happen to the training error? What might happen to the test error?

## Classical Machine Learning

*Note: Classical ML algorithms weren't detailed in the provided attachments.*

## Deep Learning Architectures and Applications

### Natural Language Processing

1. [E] What's the motivation for RNN?
2. [E] What's the motivation for LSTM?
3. [M] How would you do dropouts in an RNN?
4. [E] What's density estimation? Why do we say a language model is a density estimator?
5. [M] Language models are often referred to as unsupervised learning, but some say its mechanism isn't that different from supervised learning. What are your thoughts?
6. [M] Why do we need word embeddings?
7. [M] What's the difference between count-based and prediction-based word embeddings?
8. [H] Most word embedding algorithms are based on the assumption that words that appear in similar contexts have similar meanings. What are some of the problems with context-based word embeddings?
9. [M] Given a query Q: "The early bird gets the worm", find the two top-ranked documents according to the TF/IDF rank using the cosine similarity measure and the term set {bird, duck, worm, early, get, love}. Are the top-ranked documents relevant to the query? (Using the 5 documents provided in the question)
10. [M] Assume that document D5 goes on to tell more about the duck and the bird and mentions "bird" three times, instead of just once. What happens to the rank of D5? Is this change in the ranking of D5 a desirable property of TF/IDF? Why?
11. [E] Your client wants you to train a language model on their dataset but their dataset is very small with only about 10,000 tokens. Would you use an n-gram or a neural language model?
12. [E] For n-gram language models, does increasing the context length (n) improve the model's performance? Why or why not?
13. [M] What problems might we encounter when using softmax as the last layer for word-level language models? How do we fix it?
14. [E] What's the Levenshtein distance of the two words "doctor" and "bottle"?
15. [M] BLEU is a popular metric for machine translation. What are the pros and cons of BLEU?
16. [H] On the same test set, LM model A has a character-level entropy of 2 while LM model A has a word-level entropy of 6. Which model would you choose to deploy?
17. [M] Imagine you have to train a NER model on the text corpus A. Would you make A case-sensitive or case-insensitive?
18. [M] Why does removing stop words sometimes hurt a sentiment analysis model?
19. [M] Many models use relative position embedding instead of absolute position embedding. Why is that?
20. [H] Some NLP models use the same weights for both the embedding layer and the layer just before softmax. What's the purpose of this?

### Computer Vision

1. [M] For neural networks that work with images like VGG-19, InceptionNet, you often see a visualization of what type of features each filter captures. How are these visualizations created?
2. [M] How are your model's accuracy and computational efficiency affected when you decrease or increase its filter size?
3. [E] How do you choose the ideal filter size?
4. [M] Convolutional layers are also known as "locally connected." Explain what it means.
5. [M] When we use CNNs for text data, what would the number of channels be for the first conv layer?
6. [E] What is the role of zero padding?
7. [E] Why do we need upsampling? How to do it?
8. [M] What does a 1x1 convolutional layer do?
9. [E] What happens when you use max-pooling instead of average pooling?
10. [E] When should we use one instead of the other?
11. [E] What happens when pooling is removed completely?
12. [M] What happens if we replace a 2 x 2 max pool layer with a conv layer of stride 2?
13. [M] When we replace a normal convolutional layer with a depthwise separable convolutional layer, the number of parameters can go down. How does this happen? Give an example to illustrate this.
14. [M] Can you use a base model trained on ImageNet (image size 256 x 256) for an object classification task on images of size 320 x 360? How?
15. [H] How can a fully-connected layer be converted to a convolutional layer?
16. [H] Pros and cons of FFT-based convolution and Winograd-based convolution.

### Reinforcement Learning

1. [E] Explain the explore vs exploit tradeoff with examples.
2. [E] How would a finite or infinite horizon affect our algorithms?
3. [E] Why do we need the discount term for objective functions?
4. [E] Fill in the empty circles using the minimax algorithm. (Diagram was in the original text)
5. [M] Fill in the alpha and beta values as you traverse the minimax tree from left to right. (Diagram was in the original text)
6. [E] Given a policy, derive the reward function.
7. [M] Pros and cons of on-policy vs. off-policy.
8. [M] What's the difference between model-based and model-free? Which one is more data-efficient?

### Other Deep Learning Topics

1. [M] An autoencoder is a neural network that learns to copy its input to its output. When would this be useful?
2. [E] What's the motivation for self-attention?
3. [E] Why would you choose a self-attention architecture over RNNs or CNNs?
4. [M] Why would you need multi-headed attention instead of just one head for attention?
5. [M] How would changing the number of heads in multi-headed attention affect the model's performance?
6. [E] You want to build a classifier to predict sentiment in tweets but you have very little labeled data (say 1000). What do you do?
7. [M] What's gradual unfreezing? How might it help with transfer learning?
8. [M] How do Bayesian methods differ from the mainstream deep learning approach?
9. [M] How are the pros and cons of Bayesian neural networks compared to the mainstream neural networks?
10. [M] Why do we say that Bayesian neural networks are natural ensembles?
11. [E] What do GANs converge to?
12. [M] Why are GANs so hard to train? 