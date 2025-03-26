# Deep Learning Q&A Reference

This document provides answers to common questions about deep learning architectures, training methods, and best practices.

*Question difficulty is marked as [E] for Easy, [M] for Medium, and [H] for Hard.*

## Table of Contents

- [Natural Language Processing](#natural-language-processing)
- [Computer Vision](#computer-vision)
- [Reinforcement Learning](#reinforcement-learning)
- [Other Deep Learning Topics](#other-deep-learning-topics)
- [Training Neural Networks](#training-neural-networks)

## Natural Language Processing

### 1. RNNs

#### [E] What's the motivation for RNN?

Recurrent Neural Networks (RNNs) were motivated by the need to process sequential data while maintaining context information. Unlike feedforward networks, RNNs:
- Can handle variable-length sequences (text, speech, time series)
- Maintain a "memory" of previous inputs through recurrent connections
- Share parameters across different time steps, making them more efficient
- Can theoretically capture long-range dependencies in sequential data

This architecture makes RNNs particularly suitable for tasks requiring temporal context, such as language modeling, speech recognition, and machine translation.

#### [E] What's the motivation for LSTM?

Long Short-Term Memory (LSTM) networks were developed to address the vanishing gradient problem that affects standard RNNs. Their key motivations include:
- Solving the vanishing/exploding gradient problem during backpropagation through time
- Enabling the network to learn long-term dependencies over hundreds of time steps
- Providing explicit mechanisms (gates) to control what information to remember or forget
- Creating paths where gradients can flow for long durations without vanishing

The gating mechanisms (input, forget, and output gates) allow LSTMs to selectively update their internal state, making them much more effective at capturing long-range dependencies compared to vanilla RNNs.

#### [M] How would you do dropouts in an RNN?

Implementing dropout in RNNs requires careful consideration to avoid disrupting the recurrent connections. Best practices include:

1. **Input and output dropout**: Apply standard dropout to the inputs and outputs of the RNN layer, but not to the recurrent connections.

2. **Variational dropout**: Use the same dropout mask at each time step for a given sequence, which helps maintain consistency in the recurrent state.

3. **Recurrent dropout**: Apply dropout only to the recurrent connections, keeping the same mask across all time steps within a sequence.

4. **Zoneout**: Rather than dropping units, randomly force some recurrent units to maintain their previous values.

Example implementation in TensorFlow/Keras:
```python
from tensorflow.keras.layers import LSTM
lstm_layer = LSTM(units=128, 
                 dropout=0.2,         # Dropout for inputs
                 recurrent_dropout=0.2  # Dropout for recurrent connections
                )
```

### 2. [E] What's density estimation? Why do we say a language model is a density estimator?

**Density estimation** is the process of estimating the probability density function of a random variable based on observed data. 

A language model is called a density estimator because:
- It estimates the probability distribution over sequences of words/tokens
- It assigns a probability P(w₁, w₂, ..., wₙ) to a sequence of words
- When using autoregressive models, it estimates P(wₜ|w₁, w₂, ..., wₜ₋₁)
- The model effectively learns the underlying probability density of the language

Language models approximate the true distribution of language by learning to assign higher probabilities to likely word sequences and lower probabilities to unlikely ones, which is fundamentally a density estimation task.

### 3. [M] Language models are often referred to as unsupervised learning, but some say its mechanism isn't that different from supervised learning. What are your thoughts?

Language models occupy an interesting position between supervised and unsupervised learning:

**Arguments for unsupervised learning**:
- They don't require explicit human-labeled data
- They learn from raw text without predefined target outputs
- The objective is to model the inherent structure and patterns in language
- They can generalize to tasks not explicitly trained for

**Arguments for supervised learning**:
- The training objective is typically next-token prediction, which has clear inputs (context) and targets (next token)
- Loss functions like cross-entropy are used, similar to supervised classification
- The model is literally "supervised" by the text itself, with each token serving as a target for previous tokens
- The mathematical formulation resembles supervised learning

My view: Language modeling is **self-supervised learning**, a middle ground where the supervision signal is derived from the input data itself. The model creates its own labels from unlabeled data, making it technically supervised in mechanism but unsupervised in data requirements. This explains why language models trained with self-supervision can be so powerful - they combine the scalability of unsupervised learning with the directed learning signal of supervision.

### 4. Word embeddings

#### [M] Why do we need word embeddings?

Word embeddings are essential because they:
- Convert discrete word tokens into continuous vector spaces where semantic relationships are preserved
- Reduce dimensionality compared to one-hot encoding (e.g., from vocabulary size to 300 dimensions)
- Enable meaningful mathematical operations on words (like finding analogies)
- Allow neural networks to process words effectively
- Help models generalize across similar words (e.g., "apple" and "orange" will have similar representations)
- Enable transfer learning from large text corpora to smaller tasks

#### [M] What's the difference between count-based and prediction-based word embeddings?

**Count-based embeddings**:
- Based on co-occurrence statistics of words in corpus
- Examples include LSA, HAL, COALS, and GloVe
- Process involves counting co-occurrences then applying dimensionality reduction
- Often use SVD or similar techniques for dimension reduction
- Computationally efficient for training

**Prediction-based embeddings**:
- Learn word vectors by predicting context words
- Examples include Word2Vec (CBOW, Skip-gram), ELMo
- Usually trained with neural networks
- Often capture more semantic information
- Can be more computationally intensive to train
- Generally perform better on semantic tasks

#### [H] Most word embedding algorithms are based on the assumption that words that appear in similar contexts have similar meanings. What are some of the problems with context-based word embeddings?

Key problems include:
- **Polysemy**: Single representation for words with multiple meanings (e.g., "bank" as financial institution vs. river bank)
- **Static nature**: Each word has the same vector regardless of context
- **Window-based context** may miss long-range dependencies
- May encode **societal biases** present in the training corpus
- **Difficulty handling rare words** or out-of-vocabulary words
- Cannot capture **compositional semantics** well (meaning of phrases)
- Newer contextual embeddings (BERT, ELMo) address some of these limitations by generating dynamic embeddings based on context

### 5. TF/IDF ranking

#### [M] Given a query and a set of documents, find the top-ranked documents according to TF/IDF.

To find top-ranked documents using TF-IDF:

1. **Calculate term frequency (TF)** for each term t in document d:
   - TF(t,d) = (Number of times t appears in d) / (Total number of terms in d)

2. **Calculate inverse document frequency (IDF)** for each term t:
   - IDF(t) = log(Total number of documents / Number of documents containing t)

3. **Calculate TF-IDF score** for each term t in document d:
   - TF-IDF(t,d) = TF(t,d) × IDF(t)

4. **For each query term**, multiply its TF-IDF score in each document

5. **Sum the scores** for all query terms in each document

6. **Rank documents** by their total scores

For example, if query Q = "deep learning" and we have 3 documents:
- D1: "Deep learning models require GPUs"
- D2: "Machine learning includes deep learning"
- D3: "Learning to code is important"

We'd calculate TF-IDF for "deep" and "learning" in each document, sum these scores, and rank the documents accordingly. D2 would likely rank highest because it contains both terms.

#### [M] How document ranking changes when term frequency changes within a document?

When term frequency changes within a document:

1. **Increased term frequency**:
   - Higher TF component for that term
   - Document becomes more relevant for queries containing that term
   - Impact diminishes logarithmically (if using log-normalization for TF)
   
2. **Decreased term frequency**:
   - Lower TF component
   - Reduced relevance for that term
   
3. **Term saturation effects**:
   - Most TF-IDF implementations use sublinear scaling (often log normalization: 1 + log(TF))
   - This dampens the effect of high frequency terms, preventing them from dominating
   - A term appearing 100 times won't be 10x more important than one appearing 10 times

4. **Document length normalization**:
   - Longer documents naturally have higher raw term frequencies
   - Normalization (dividing by document length) prevents bias toward longer documents

This is why modern search engines use more sophisticated variants of TF-IDF that account for these effects, such as BM25, which explicitly handles term saturation and document length normalization.

### 6. [E] Your client wants you to train a language model on their dataset but their dataset is very small with only about 10,000 tokens. Would you use an n-gram or a neural language model?

For a small dataset of only 10,000 tokens, an **n-gram language model** would be more appropriate than a neural language model, for several reasons:

- **Data efficiency**: N-gram models require less data to estimate probabilities reliably
- **Overfitting risk**: Neural models have many parameters and would likely overfit severely on such a small dataset
- **Simplicity**: N-gram models are simpler and faster to train
- **Smoothing techniques**: N-gram models can use techniques like Kneser-Ney smoothing to handle sparse data
- **Interpretability**: N-gram models are more interpretable, which may be valuable for a client project

However, I would recommend additional approaches:
- Use transfer learning if possible (fine-tune a pre-trained neural LM)
- Apply strong regularization if using neural models
- Consider data augmentation techniques to expand the dataset
- Limit vocabulary size to reduce sparsity

### 7. [E] For n-gram language models, does increasing the context length (n) improve the model's performance? Why or why not?

Increasing the context length (n) in n-gram models has both advantages and disadvantages:

**Advantages**:
- Captures longer dependencies and patterns
- Models more complex language structures
- Can represent more specific contexts

**Disadvantages**:
- Suffers from data sparsity (many n-grams never appear in training)
- Requires exponentially more data as n increases
- Increases storage requirements dramatically
- May lead to overfitting on training data

In practice, performance typically improves as n increases from 1 to around 5, then plateaus or degrades for higher values due to sparsity issues. The optimal value depends on:
- Dataset size (larger datasets can support larger n)
- Application requirements
- Available computational resources

Modern approaches often use backoff or interpolation methods that combine multiple n-gram orders to get the benefits of higher-order models while mitigating sparsity issues.

### 8. [M] What problems might we encounter when using softmax as the last layer for word-level language models? How do we fix it?

Using softmax in word-level language models presents several challenges:

**Problems**:
1. **Computational cost**: Computing softmax over large vocabularies (often 50K-1M words) is expensive
2. **Memory usage**: Requires storing parameters for every word in vocabulary
3. **Rare words**: Difficult to learn good representations for infrequent words
4. **Out-of-vocabulary words**: Cannot handle words not seen during training

**Solutions**:
1. **Hierarchical softmax**: Organizes vocabulary in a tree structure, reducing complexity from O(V) to O(log V)
2. **Sampled softmax**: Only computes probabilities for the correct word and a small sample of incorrect words during training
3. **Noise Contrastive Estimation (NCE)**: Transforms the problem into binary classification between true and noise samples
4. **Adaptive softmax**: Uses different capacity for frequent vs. rare words
5. **Character/subword tokenization**: Breaks words into smaller units (WordPiece, BPE, SentencePiece)
6. **Self-normalization techniques**: Train the model to produce approximately normalized scores without explicit normalization

Most modern language models use a combination of these approaches, particularly subword tokenization combined with sampled softmax during training.

### 9. [E] What's the Levenshtein distance of the two words "doctor" and "bottle"?

The Levenshtein distance between "doctor" and "bottle" is **5**.

Let's calculate it step by step:

1. Levenshtein distance measures the minimum number of single-character edits (insertions, deletions, substitutions) required to change one word into another.

2. Starting with "doctor" to transform it to "bottle":
   - Replace 'd' with 'b' (substitution): "boctor" (1 operation)
   - Replace 'o' with 'o' (no change): "boctor" (still 1 operation)
   - Replace 'c' with 't' (substitution): "bottor" (2 operations)
   - Replace 't' with 't' (no change): "bottor" (still 2 operations)
   - Replace 'o' with 'l' (substitution): "bottlr" (3 operations)
   - Replace 'r' with 'e' (substitution): "bottle" (4 operations)

However, the true minimum edit distance is actually 5, as we can trace using the dynamic programming approach:

|       | ∅ | b | o | t | t | l | e |
|-------|---|---|---|---|---|---|---|
| ∅     | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| d     | 1 | 1 | 2 | 3 | 4 | 5 | 6 |
| o     | 2 | 2 | 1 | 2 | 3 | 4 | 5 |
| c     | 3 | 3 | 2 | 2 | 3 | 4 | 5 |
| t     | 4 | 4 | 3 | 2 | 2 | 3 | 4 |
| o     | 5 | 5 | 4 | 3 | 3 | 3 | 4 |
| r     | 6 | 6 | 5 | 4 | 4 | 4 | 4 |

Following the dynamic programming matrix, the Levenshtein distance is 5.

### 10. [M] BLEU is a popular metric for machine translation. What are the pros and cons of BLEU?

**BLEU (Bilingual Evaluation Understudy)** is a metric for evaluating machine translation quality by comparing generated translations to reference translations.

**Pros**:
- **Automated and efficient**: No human evaluation required
- **Language-independent**: Works across different language pairs
- **Correlates with human judgment** at the corpus level
- **Simple to understand**: Based on n-gram precision
- **Industry standard**: Enables comparison across different systems
- **Fast computation**: Can be calculated quickly for large test sets

**Cons**:
- **Focuses on precision, not recall**: Doesn't penalize missing content adequately
- **No semantic understanding**: Purely lexical matching misses meaning equivalence
- **Poor sentence-level correlation** with human judgments
- **Insensitive to grammaticality**: Doesn't well capture fluency and structure
- **Reference bias**: Heavily dependent on reference translation style and count
- **Penalizes valid paraphrasing**: Different but correct translations score poorly
- **No consideration of importance**: All n-grams weighted equally regardless of significance

Modern MT evaluation often supplements BLEU with other metrics like METEOR, chrF, TER, or BERTScore which address some of these limitations.

### 11. [H] On the same test set, LM model A has a character-level entropy of 2 while LM model B has a word-level entropy of 6. Which model would you choose to deploy?

This question requires careful analysis as we're comparing different granularity levels:

**Key insight**: Character-level and word-level entropies cannot be directly compared. We need to normalize them to the same units.

For English text:
- Average word length is ~5 characters (including spaces)
- Character-level entropy measures bits per character
- Word-level entropy measures bits per word

**Conversion**:
- Model A: 2 bits/character × ~5 characters/word = ~10 bits/word
- Model B: 6 bits/word

Therefore, Model B (6 bits/word) has lower normalized entropy than Model A (~10 bits/word), suggesting it has a better understanding of the language structure.

**Conclusion**: Deploy Model B, as it provides a more compact (and likely more accurate) representation of the language probability distribution. However, consider other factors:

- Deployment constraints (model size, inference speed)
- Task requirements (some applications may benefit from character-level models)
- Out-of-vocabulary handling capabilities
- Additional metrics beyond entropy (perplexity on domain-specific text)

### 12. [M] Imagine you have to train a NER model on the text corpus A. Would you make A case-sensitive or case-insensitive?

For Named Entity Recognition (NER), I would generally **maintain case sensitivity** in the corpus for the following reasons:

**Advantages of case-sensitive NER**:
- Capitalization provides strong signals for entity detection (e.g., "Apple" company vs. "apple" fruit)
- Proper nouns are typically capitalized in many languages
- Acronyms rely on case information (NASA vs. nasa)
- State-of-the-art NER systems typically use case information
- Preserves more information from the original text

**However, considerations for case-insensitivity**:
- If the corpus contains a lot of noisy text (social media, informal communications) with inconsistent capitalization
- For languages with limited or no case distinctions
- When working with speech transcripts that may lack proper capitalization
- If the corpus is very small and case variants would fragment the limited training data

**Best approach**: A hybrid solution where the model receives both:
1. The original case-sensitive word
2. Case information as an explicit feature (e.g., all caps, title case, lowercase)

This way, the model can learn when case is important for entity detection and when it might be misleading. Most modern NER systems like SpaCy, BERT-based NER, and BiLSTM-CRF models maintain case sensitivity while being robust to case variations.

### 13. [M] Why does removing stop words sometimes hurt a sentiment analysis model?

Removing stop words can be detrimental to sentiment analysis for several important reasons:

1. **Negations lost**: Words like "not", "no", "never" are crucial for sentiment but often classified as stop words (e.g., "not good" vs "good")

2. **Intensity modifiers removed**: Terms like "very", "extremely", "somewhat" indicate sentiment intensity

3. **Contextual meaning disrupted**: Stop words provide grammatical structure that helps resolve ambiguity

4. **Question/statement distinction blurred**: Removing question words changes the interpretive framework

5. **Idiomatic expressions broken**: Many sentiment-bearing phrases include stop words (e.g., "over the moon")

6. **Modern models don't need filtering**: Neural models like BERT learn contextual representations where stop words contribute meaningful signal

7. **Comparative constructions affected**: Phrases like "better than" or "worse than" lose meaning

Example demonstrating the issue:
- Original: "This movie is not good at all"
- Without stop words: "movie good"
- The sentiment flips from negative to positive

Best practice: For modern sentiment analysis using neural networks, retain all words and let the model learn which ones matter for the task.

### 14. [M] Many models use relative position embedding instead of absolute position embedding. Why is that?

Relative position embeddings have gained popularity over absolute position embeddings for several compelling reasons:

**Advantages of relative position embeddings**:

1. **Length generalization**: Models can generalize to sequences longer than those seen during training

2. **Translation invariance**: Patterns learned at one position can be recognized at other positions

3. **Locality bias**: They naturally emphasize relationships between nearby tokens, which aligns with linguistic structures

4. **Parameter efficiency**: Can be more parameter-efficient as they encode relationships rather than absolute positions

5. **Better inductive bias**: Words often have similar relationships regardless of where they appear in a sentence

6. **Improved attention mechanisms**: In Transformers, relative position information can be directly incorporated into attention calculations (as in Shaw et al., 2018 and Transformer-XL)

7. **Hierarchical structures**: Better captures hierarchical relationships in language where relative positions matter more than absolute ones

Models like Transformer-XL, Music Transformer, and certain BERT variants use relative positional encodings and show improved performance on tasks requiring understanding of long-range dependencies and generalization to longer sequences than seen during training.

### 15. [H] Some NLP models use the same weights for both the embedding layer and the layer just before softmax. What's the purpose of this?

**Weight tying** (sharing weights between input embeddings and pre-softmax layers) serves several important purposes:

1. **Parameter reduction**: Dramatically decreases model size by eliminating redundant parameters (especially important for large vocabulary models)

2. **Regularization effect**: Acts as a form of regularization, reducing overfitting and improving generalization

3. **Semantic consistency**: Enforces consistency between how words are represented as inputs and outputs

4. **Theoretical motivation**: For language modeling, input and output representations should ideally capture the same semantic space

5. **Improved gradient flow**: Creates a direct path for gradients to flow between output and input layers

6. **Performance gains**: Empirically shown to improve perplexity in language models (Press & Wolf, 2017; Inan et al., 2017)

7. **Learning efficiency**: Often leads to faster convergence as the model learns a unified representation

This technique is common in models like AWD-LSTM, Transformer-based language models, and various neural machine translation architectures. The shared weights create a more coherent representation space where the meaning of a word is consistent whether it's being predicted or used as context.

## Computer Vision

### 1. [M] For neural networks that work with images like VGG-19, InceptionNet, you often see a visualization of what type of features each filter captures. How are these visualizations created?

Filter visualizations in CNNs can be created through several techniques:

1. **Activation Maximization**:
   - Start with a random noise image
   - Perform gradient ascent on the input to maximize activation of a specific filter
   - Constrain optimization to produce natural-looking images (e.g., using regularization)
   - Result shows patterns that maximally activate the filter

2. **Deconvolutional Network Approach** (Zeiler & Fergus):
   - Pass an image forward through the network
   - Select a specific activation in a layer
   - Zero out all other activations
   - Reverse the network operations (deconvolution, unpooling) to project back to pixel space
   - Shows which input patterns caused specific activations

3. **Guided Backpropagation**:
   - Similar to deconvnet but modifies the backpropagation to only allow positive gradients through ReLU layers
   - Produces cleaner visualizations

4. **Feature Inversion**:
   - Start with a feature representation at a specific layer
   - Generate an image that would produce similar feature activations
   - Shows what visual information is preserved at different layers

5. **Class Activation Mapping (CAM) and Grad-CAM**:
   - Identifies important regions in an image for a specific class prediction
   - Useful for understanding what parts of an image influence classification decisions

These visualizations reveal how networks progress from detecting simple edges and textures in early layers to more complex shapes, parts, and objects in deeper layers.

### 2. Filter size

#### [M] How are your model's accuracy and computational efficiency affected when you decrease or increase its filter size?

**Increasing filter size**:

*Effects on accuracy*:
- Captures larger spatial patterns and contextual information
- Better for detecting large-scale features
- May help with understanding global structure in images
- Can reduce aliasing effects

*Effects on computational efficiency*:
- Quadratic increase in parameters (O(k²) where k is filter size)
- More FLOPs required per convolution
- Slower inference and training
- Higher memory requirements

**Decreasing filter size**:

*Effects on accuracy*:
- Better for capturing fine details and textures
- May miss larger patterns without sufficient depth
- Often requires more layers to achieve equivalent receptive field
- May introduce aliasing if too small

*Effects on computational efficiency*:
- Fewer parameters per layer
- Faster computation per layer
- Lower memory footprint per layer
- May require more layers to achieve similar performance

Modern architectures often use small filters (3×3 or even 1×1) in deeper networks, as stacking multiple small filters:
1. Achieves similar receptive fields as larger filters
2. Introduces more non-linearities (ReLU after each conv)
3. Reduces parameter count for equivalent receptive field
4. Often results in better performance/computation trade-off

#### [E] How do you choose the ideal filter size?

Choosing the ideal filter size involves considering several factors:

1. **Nature of features**: Match filter size to feature scale
   - Small filters (1×1, 3×3): Fine details, textures, efficiency
   - Medium filters (5×5, 7×7): Moderate patterns, balance
   - Large filters (9×9+): Global structures, but rarely used in deep layers

2. **Network depth**: 
   - Deeper networks can use smaller filters (3×3 is common)
   - First layer often uses larger filters (e.g., 7×7 in ResNet) to capture initial patterns

3. **Computational constraints**:
   - Smaller filters = fewer parameters = less computation
   - Stacking small filters often more efficient than one large filter

4. **Modern best practices**:
   - Use multiple 3×3 filters instead of a single large filter
   - Mix filter sizes within network (e.g., Inception modules)
   - Use 1×1 convolutions for dimensionality reduction

5. **Empirical validation**:
   - Test different sizes on validation data
   - Consider accuracy vs. computational trade-off
   - Use architecture search if resources permit

Most state-of-the-art CNNs favor small filters (especially 3×3) throughout most of the network with occasional 1×1 filters for channel-wise operations, which offers a good balance of expressiveness and efficiency.

### 3. [M] Convolutional layers are also known as "locally connected." Explain what it means.

Convolutional layers are called "locally connected" because of their distinctive connectivity pattern:

1. **Spatial locality**: Each neuron only connects to a small region (receptive field) of the input, not the entire input
   
2. **Parameter sharing**: The same set of weights is applied across different spatial locations, unlike fully connected layers where each connection has a unique weight

3. **Connectivity visualization**:
   - In fully connected layers: Each output neuron connects to ALL input neurons
   - In convolutional layers: Each output neuron connects ONLY to input neurons within its receptive field

4. **Key properties of local connectivity**:
   - Preserves spatial relationships in the data
   - Dramatically reduces parameter count compared to fully connected layers
   - Creates translation invariance (ability to detect features regardless of location)
   - Enforces a locality bias that aligns with natural image statistics

5. **Example**: In a 3×3 convolution, each output pixel depends only on a 3×3 region of the input, not the entire image

This local connectivity is inspired by the visual cortex in mammals, where neurons respond to stimuli only in a limited region of the visual field. It enables CNNs to efficiently learn hierarchical patterns while maintaining spatial awareness.

### 4. [M] When we use CNNs for text data, what would the number of channels for the first conv layer be?

When applying CNNs to text data, the number of channels in the first convolutional layer depends on the word embedding dimension:

**Number of channels = Embedding dimension**

Explanation:
1. Text is typically represented as a 2D matrix:
   - Rows: Words/tokens in the sequence (e.g., sentence length)
   - Columns: Dimensions of the word embedding (e.g., 300 for GloVe)

2. Unlike images (which have 1 or 3 input channels), each "pixel" in text data is a multi-dimensional vector:
   - RGB image: 3 channels (R, G, B values)
   - Text data: d channels (embedding dimensions) per word

3. The first convolutional layer receives:
   - Input shape: (batch_size, sequence_length, embedding_dim)
   - Treats embedding_dim as channels
   - Convolves along the sequence dimension

4. Common embedding dimensions:
   - Word2Vec/GloVe: 100-300
   - FastText: 300
   - BERT: 768
   - GPT-2: 768-1600

Example architecture:
```python
model = Sequential([
    Embedding(vocab_size, 300, input_length=max_sequence_length),  # 300 channels
    Conv1D(filters=128, kernel_size=3, activation='relu'),  # 1D convolution along sequence
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])
```

For character-level CNNs, the channel dimension would be the character embedding size, which might be smaller (e.g., 16-64).

### 5. [E] What is the role of zero padding?

Zero padding serves several crucial roles in convolutional neural networks:

1. **Preserving spatial dimensions**:
   - Without padding, convolutions reduce output size with each layer
   - "Same" padding maintains spatial dimensions, allowing for deeper networks

2. **Border information preservation**:
   - Without padding, pixels at the edges would be used less frequently
   - Padding ensures border information contributes equally to feature maps

3. **Control over spatial reduction**:
   - Allows architects to precisely control when and how spatial dimensions decrease
   - Enables clean network designs (e.g., maintaining power-of-2 dimensions)

4. **Deep network construction**:
   - Enables building very deep networks without premature reduction of feature maps
   - Critical for architectures like ResNet and DenseNet

5. **Receptive field management**:
   - Helps manage the growth of the effective receptive field
   - Balances global vs. local information processing

Types of padding:
- **Valid padding** (no padding): Output shrinks with each layer
- **Same padding**: Output dimensions match input dimensions
- **Full padding**: Output dimensions larger than input (rare)

Zero is used as the padding value because it minimally impacts convolution results near the borders, though other padding strategies (reflection, replication) can sometimes perform better for specific tasks.

### 6. [E] Why do we need upsampling? How to do it?

Upsampling is essential in deep learning for several key purposes:

**Why upsampling is needed**:
1. **Generating higher resolution outputs**: For tasks like image super-resolution or generation
2. **Encoder-decoder architectures**: To restore spatial dimensions reduced by pooling/striding in encoder
3. **Semantic segmentation**: For pixel-wise classification at original resolution
4. **Feature map enlargement**: To match spatial dimensions for feature fusion

**Common upsampling methods**:

1. **Nearest Neighbor**:
   - Simplest approach; repeats pixels
   - Fast but creates blocky artifacts
   - No learnable parameters

2. **Bilinear/Bicubic Interpolation**:
   - Smoother than nearest neighbor
   - Deterministic, no learning required
   - Better preservation of details than nearest neighbor

3. **Transposed Convolution** (Deconvolution):
   - Learnable upsampling operation
   - Can create checkerboard artifacts if not carefully designed
   - Kernel determines how pixels get expanded

4. **Pixel Shuffle** (Sub-pixel Convolution):
   - Rearranges elements from feature maps
   - Efficient upsampling without artifacts
   - Used in super-resolution networks (ESPCN)

5. **Unpooling**:
   - Reverses max-pooling by placing values at recorded positions
   - Preserves structural information better than simple interpolation

Modern architectures often combine upsampling with regular convolutions (e.g., resize-convolution) to reduce artifacts while maintaining learnable parameters for optimal reconstruction.

### 7. [M] What does a 1x1 convolutional layer do?

A 1×1 convolutional layer, also known as a "network in network" or "pointwise convolution," serves several important functions:

1. **Dimensionality reduction/expansion**:
   - Can reduce or increase the number of channels
   - Acts as a learned linear projection along the channel dimension
   - Helps control model capacity and computational cost

2. **Cross-channel interactions**:
   - Allows information to flow between channels at each spatial location
   - Creates new features that are linear combinations of input channels
   - Adds non-linearity across channels when followed by activation functions

3. **Bottleneck architecture**:
   - Used to reduce channels before applying expensive 3×3 or 5×5 convolutions
   - Then expand channels back afterward
   - Drastically reduces computation (e.g., in ResNet, Inception, MobileNet)

4. **Network regularization**:
   - Introduces additional non-linearities without changing spatial dimensions
   - Can help prevent overfitting by reducing model capacity

5. **Implementation efficiency**:
   - Computationally inexpensive compared to larger convolutions
   - Efficiently implemented as matrix multiplication

Example in bottleneck block:
```
Input (256 channels) → 1×1 Conv (64 channels) → 3×3 Conv (64 channels) → 1×1 Conv (256 channels)
```

This pattern is fundamental to many efficient architectures like ResNet, Inception, and modern mobile-optimized networks.

### 8. Pooling

#### [E] What happens when you use max-pooling instead of average pooling?

When using max-pooling instead of average pooling:

**Effects of max-pooling**:
- **Feature selection**: Preserves the strongest activations/features
- **Invariance to small translations**: More robust to slight positional changes
- **Sparse representations**: Encourages sparse, high-activation features
- **Sharp feature detection**: Better at detecting distinct, pronounced features
- **Preserves texture details**: Maintains high-frequency information
- **Noise amplification**: Can amplify noise if it produces high activations

**Effects of average pooling**:
- **Feature aggregation**: Considers all values in the pooling window
- **Smoother downsampling**: Produces more stable, smoother feature maps
- **Background preservation**: Better retains information from all parts of the feature map
- **Noise reduction**: Naturally dampens noisy activations
- **Global context**: When used globally, provides overall image statistics

Visual differences:
- Max-pooling tends to highlight edges, textures, and distinctive patterns
- Average pooling tends to highlight broader, more distributed features

These differences explain why max-pooling is usually preferred in early/mid layers of classification networks, while average pooling is often used for global feature aggregation in the final layers.

#### [E] When should we use one instead of the other?

**When to use max-pooling**:
- **Classification tasks**: Where distinctive features matter most
- **When detecting presence** of features is important (object detection, classification)
- **Early and middle layers** of deep networks
- **When working with sparse features** that have many near-zero activations
- **Tasks requiring translation invariance**
- **When preserving texture details** is important

**When to use average pooling**:
- **Global feature aggregation** (Global Average Pooling layers before classification)
- **When smoothness is desired** (segmentation, generation tasks)
- **When all values in a region** are informative (not just peaks)
- **Reducing spatial dimensions** while preserving energy distribution
- **When working with dense features** where all values carry meaning
- **Tasks requiring positional stability**
- **Noise-sensitive applications**

**Hybrid approaches**:
- Some architectures use max-pooling in early layers and average pooling later
- BlurPool and other learned pooling operations can combine benefits
- Weighted average pooling adapts based on the importance of each value

Best practice is to experiment with both for your specific task and architecture, as performance differences can be task-dependent.

#### [E] What happens when pooling is removed completely?

Removing pooling layers from CNNs has several significant effects:

**Architecture impacts**:
- **Maintained spatial dimensions**: No reduction in height/width throughout the network
- **Computational cost increase**: More operations required without downsampling
- **Memory usage increase**: Larger feature maps throughout the network
- **Increased parameter count**: If pooling is replaced by strided convolutions

**Performance effects**:
- **Higher spatial resolution**: Preserves fine-grained spatial information
- **Reduced translation invariance**: Less robust to small positional shifts
- **More detailed features**: Can capture finer details and preserve spatial relationships
- **Potential for overfitting**: Higher capacity model may overfit on small datasets
- **Gradient flow**: Potentially better gradient flow without pooling's information bottleneck

**Modern alternatives**:
- **Strided convolutions**: Replace pooling with stride>1 convolutions (used in many GANs)
- **Dilated/atrous convolutions**: Expand receptive field without pooling
- **Self-attention mechanisms**: Capture long-range dependencies without pooling

Networks like All-Convolutional Net demonstrated competitive performance without pooling layers by using strided convolutions instead. Modern architectures often use a mix of approaches, with some designs minimizing or eliminating traditional pooling in favor of learned downsampling operations.

#### [M] What happens if we replace a 2 x 2 max pool layer with a conv layer of stride 2?

Replacing a 2×2 max-pooling layer with a convolutional layer with stride 2 results in several important changes:

**Key differences**:

1. **Learnable downsampling**:
   - Pooling: Fixed operation (always takes maximum/average)
   - Strided conv: Learns optimal downsampling weights

2. **Feature preservation**:
   - Pooling: Discards 75% of values (in 2×2 case), keeping only maxima
   - Strided conv: All values contribute to output, weighted by learned filters

3. **Parameter count**:
   - Pooling: Zero parameters (parameter-free)
   - Strided conv: k²×Cin×Cout parameters (k=kernel size)

4. **Computation**:
   - Pooling: Simple max operation, computationally efficient
   - Strided conv: More computationally intensive

5. **Invariance properties**:
   - Pooling: Built-in local translation invariance
   - Strided conv: Must learn translation invariance patterns

6. **Information flow**:
   - Pooling: Creates information bottleneck, discards information
   - Strided conv: Can selectively retain information via learned weights

This replacement approach has been used successfully in "all-convolutional" networks and GANs (which often avoid pooling). Research has shown strided convolutions can match or exceed pooling performance when properly trained, though they require more computation and parameters.

### 9. [M] When we replace a normal convolutional layer with a depthwise separable convolutional layer, the number of parameters can go down. How does this happen? Give an example to illustrate this.

Depthwise separable convolutions dramatically reduce parameters by factorizing a standard convolution into two simpler operations:

**Standard convolution**:
- Applies a k×k filter across all input channels simultaneously
- Parameters: k×k×Cin×Cout

**Depthwise separable convolution**:
1. **Depthwise convolution**: Applies separate k×k filter to each input channel
   - Parameters: k×k×Cin
2. **Pointwise convolution**: 1×1 convolution to combine filtered channels
   - Parameters: 1×1×Cin×Cout = Cin×Cout

**Total parameters**: k×k×Cin + Cin×Cout

**Parameter reduction ratio**: (k×k×Cin + Cin×Cout) / (k×k×Cin×Cout) = (1/Cout + 1/k²)

**Example**:
For a 3×3 convolution with 128 input channels and 256 output channels:

- **Standard convolution**:
  3×3×128×256 = 294,912 parameters

- **Depthwise separable**:
  - Depthwise: 3×3×128 = 1,152 parameters
  - Pointwise: 128×256 = 32,768 parameters
  - Total: 33,920 parameters

- **Reduction**: ~8.7× fewer parameters (294,912 vs 33,920)

This dramatic efficiency is why models like MobileNet, XceptionNet, and EfficientNet use depthwise separable convolutions as their primary building block, achieving comparable accuracy with far fewer parameters and computation.

### 10. [M] Can you use a base model trained on ImageNet (image size 256 x 256) for an object classification task on images of size 320 x 360? How?

Yes, a base model trained on ImageNet with 256×256 images can be adapted for 320×360 images through several approaches:

**Method 1: Input Adaptation**
- **Resize images**: Scale input images to 256×256 before feeding to the model
  - Pros: Simplest approach, no model modification needed
  - Cons: May lose information or distort aspect ratio

- **Center crop**: Take a 256×256 center crop from the larger images
  - Pros: Maintains scale but loses edge information
  - Cons: May discard important features at the edges

**Method 2: Model Adaptation**
- **Replace input layer**: Keep all weights except the first layer, which is adjusted for larger input
  - Pros: Maintains most pre-trained weights
  - Cons: First layer needs retraining

- **Fully Convolutional Network conversion**:
  - Replace final fully-connected layers with convolutional or global pooling layers
  - Convolutional layers naturally handle any input size
  - Pros: Maintains spatial information throughout network
  - Cons: May require reshaping/adapting the classification head

**Method 3: Advanced Techniques**
- **Spatial Pyramid Pooling (SPP)**:
  - Add SPP layer before fully-connected layers
  - Outputs fixed-length vectors regardless of input size
  - Pros: Handles variable input sizes elegantly
  - Cons: Requires modifying network architecture

Best practice is typically to:
1. Remove the final fully-connected layers
2. Add global pooling after the last convolutional layer
3. Add new classification layers for your specific task
4. Fine-tune the entire network or just the new layers on your data

This approach works well for transfer learning regardless of input size differences.

### 11. [H] How can a fully-connected layer be converted to a convolutional layer?

A fully-connected layer can be converted to an equivalent convolutional layer through the following transformation:

**Mathematical equivalence**:
- A fully-connected layer with input size n and output size m is equivalent to:
- A convolutional layer with kernel size equal to the input spatial dimensions, and m filters

**Step-by-step conversion**:
1. For an FC layer connecting a flattened feature map of size h×w×c to m outputs:
   - Replace with a convolution layer with kernel size (h,w), c input channels, m output channels, and stride=1
   - Reshape the FC weights of shape (h×w×c, m) to (h, w, c, m)

2. For a stack of FC layers:
   - Convert the first FC layer as above
   - Convert subsequent FC layers to 1×1 convolutions

**Example**:
- Input feature map: 7×7×512
- FC layer with 4096 outputs
- Converted to: Conv with kernel_size=7×7, in_channels=512, out_channels=4096

**Advantages**:
- Network can now process inputs of any size larger than the original input
- Maintains spatial information throughout the network
- Enables dense prediction (semantic segmentation, object detection)
- Allows sliding window implementation with shared computation

This technique is fundamental to modern computer vision frameworks like FCN (Fully Convolutional Networks) for semantic segmentation, where classification networks pretrained on fixed-size inputs are converted to handle arbitrary-sized images and produce dense predictions.

### 12. [H] Pros and cons of FFT-based convolution and Winograd-based convolution.

**FFT-based Convolution**

*Pros*:
- **Asymptotic efficiency**: O(n log n) complexity versus O(n²) for direct convolution
- **Optimal for large kernels**: More efficient as kernel size increases
- **Well-suited for large feature maps**: Performance advantage grows with feature map size
- **Hardware optimized**: Highly optimized FFT libraries available on most platforms
- **Deterministic performance**: Runtime is consistent regardless of data content

*Cons*:
- **Memory overhead**: Requires additional memory for FFT computations
- **Padding issues**: Needs zero-padding to prevent circular convolution effects
- **Less efficient for small kernels**: Overhead may exceed benefits for 3×3 kernels
- **Complex number arithmetic**: More computationally intensive per operation
- **Limited batch processing efficiency**: Less efficient for small batch sizes

**Winograd-based Convolution**

*Pros*:
- **Minimal multiplications**: Reduces multiplication operations significantly
- **Optimal for small kernels**: Especially efficient for common 3×3 kernels
- **Low memory overhead**: Requires less additional memory than FFT
- **Computationally efficient**: Often 2-3× faster than direct convolution for 3×3 filters
- **Well-suited for modern CNN architectures**: Optimized for popular kernel sizes

*Cons*:
- **Limited kernel sizes**: Efficiency gains primarily for small, odd-sized kernels
- **Numerical precision issues**: More sensitive to numerical stability problems
- **Complex implementation**: More difficult to implement and optimize
- **Transformation overhead**: Pre/post-processing transformations add compute cost
- **Less advantage for large kernels**: Benefits diminish as kernel size increases

**Practical usage**:
- Modern frameworks like cuDNN use hybrid approaches:
  - Winograd for small kernels (3×3, 5×5)
  - FFT for medium kernels
  - Direct convolution for 1×1 and very large kernels
- Hardware-specific optimizations often determine the best method for specific layer dimensions

## Reinforcement Learning

### 1. [E] Explain the explore vs exploit tradeoff with examples.

The exploration vs. exploitation tradeoff is a fundamental concept in reinforcement learning:

**Core concept**: Balancing between trying new actions to discover potentially better rewards (exploration) and choosing known high-reward actions (exploitation).

**Examples in different domains**:

1. **Multi-armed bandit problem**:
   - **Scenario**: Casino with multiple slot machines, each with unknown payout probabilities
   - **Exploitation**: Keep playing the machine that has given highest rewards so far
   - **Exploration**: Try different machines to find one with potentially higher payouts
   - **Tradeoff**: More exploration means potentially missing short-term rewards; more exploitation might miss discovering the optimal machine

2. **Restaurant choice**:
   - **Exploitation**: Return to your favorite restaurant where you know the food is good
   - **Exploration**: Try a new restaurant that might be even better
   - **Tradeoff**: Comfort and reliability vs. potential for discovering a new favorite

3. **Recommendation systems**:
   - **Exploitation**: Recommend items similar to what the user has liked before
   - **Exploration**: Recommend novel items to learn more about user preferences
   - **Tradeoff**: User satisfaction now vs. improving recommendations long-term

**Common strategies**:
- **ε-greedy**: Choose best known action with probability 1-ε, random action with probability ε
- **Upper Confidence Bound (UCB)**: Select actions based on upper bound of confidence interval
- **Thompson Sampling**: Choose actions according to their probability of being optimal
- **Boltzmann exploration**: Probabilistic selection based on expected rewards

The optimal balance typically shifts from more exploration early in learning to more exploitation later as knowledge improves.

### 2. [E] How would a finite or infinite horizon affect our algorithms?

The horizon (finite vs. infinite) significantly impacts reinforcement learning algorithms:

**Finite Horizon Effects**:
- **Time-dependent policies**: Optimal action may change based on remaining time steps
- **Backward induction**: Can solve exactly using dynamic programming from the end state
- **Decreasing emphasis on future rewards**: Actions near the end focus more on immediate rewards
- **Explicit deadline awareness**: Agent behaves differently as deadline approaches
- **Value functions include time**: V(s,t) or Q(s,a,t) where t is time step
- **Examples**: Game with fixed turns, portfolio optimization until retirement date

**Infinite Horizon Effects**:
- **Stationary policies**: Optimal action depends only on state, not time
- **Discount factor necessity**: Must discount future rewards (γ<1) for convergence
- **Recurrent state emphasis**: Focus on recurring states that matter long-term
- **Value iteration/policy iteration**: Typical solution methods
- **Simpler value functions**: V(s) or Q(s,a) without time component
- **Examples**: Continuous control problems, ongoing customer interactions

**Algorithm adjustments**:
1. **Finite horizon**:
   - Dynamic programming with explicit time tracking
   - Terminal state rewards
   - Backward induction through value function

2. **Infinite horizon**:
   - Discount factor to ensure convergence
   - Focus on finding stationary policies
   - Value or policy iteration until convergence

3. **Approximation**:
   - Many infinite horizon problems are approximated with large finite horizons
   - "Effective horizon" is ~1/(1-γ) time steps for discount factor γ

Practical systems often use infinite horizon formulations with discounting as they're more computationally tractable and generalize better in continuing tasks.

### 3. [E] Why do we need the discount term for objective functions?

The discount factor (γ) in reinforcement learning objective functions serves several essential purposes:

1. **Mathematical convergence**:
   - Ensures the sum of rewards converges to a finite value in infinite-horizon problems
   - Without discounting, the total reward could be infinite, making optimization impossible

2. **Uncertainty handling**:
   - Reflects increasing uncertainty about far-future rewards
   - Future states are less predictable due to environment stochasticity

3. **Present value economics**:
   - Models time preference (reward now is worth more than the same reward later)
   - Aligns with economic and decision theory principles

4. **Computational benefits**:
   - Creates a "soft horizon" - events beyond ~1/(1-γ) steps have minimal impact
   - Helps value iteration and policy iteration algorithms converge faster

5. **Avoids policy oscillation**:
   - Stabilizes learning by preventing cycles in policy search
   - Reduces sensitivity to small changes in reward structure

6. **Risk aversion modeling**:
   - Higher discount rates (smaller γ) model risk-averse behavior
   - Lower discount rates (larger γ) encourage long-term planning

**Practical considerations**:
- γ typically ranges from 0.9 to 0.999 depending on the task
- γ close to 1: Far-sighted agent, considers distant future rewards
- γ close to 0: Myopic agent, prioritizes immediate rewards
- For episodic tasks with clear endpoints, sometimes γ=1 is used (no discounting)

The choice of discount factor is a crucial hyperparameter that balances short-term reward maximization against long-term strategic planning.

### 4. [E] Fill in the empty circles using the minimax algorithm (for a given game tree).

To fill in empty circles in a minimax game tree:

**Minimax Algorithm Steps**:
1. **Leaf node evaluation**: Assign values to all leaf nodes (terminal states)
2. **Bottom-up traversal**: Work upward from leaves to root
3. **MAX nodes** (typically represented by squares): Choose maximum child value
4. **MIN nodes** (typically represented by circles): Choose minimum child value

**Example Process**:
Assume we have this partial game tree with some values known:
```
       □ (Root/MAX)
      /|\
     / | \
    ○  ○  ○ (MIN)
   /|\ /|\ /|\
  □ □ □ □ □ □ □ □ □ (MAX)
  3 8 2 5 9 1 7 4 6 (Leaf values)
```

**Working bottom-up**:
1. First level of MIN nodes (circles):
   - Left circle = min(3,8,2) = 2
   - Middle circle = min(5,9,1) = 1
   - Right circle = min(7,4,6) = 4

2. Root MAX node (square):
   - Root = max(2,1,4) = 4

**Final tree**:
```
       □ 4
      /|\
     / | \
    ○2 ○1 ○4
   /|\ /|\ /|\
  □ □ □ □ □ □ □ □ □
  3 8 2 5 9 1 7 4 6
```

The minimax algorithm guarantees the optimal strategy assuming that both players play perfectly, with the MAX player trying to maximize their score and the MIN player trying to minimize MAX's score.

### 5. [M] Fill in the alpha and beta values as you traverse the minimax tree from left to right (for alpha-beta pruning).

In alpha-beta pruning, we maintain two values:
- **Alpha**: Best already-explored option for MAX player (initially -∞)
- **Beta**: Best already-explored option for MIN player (initially +∞)

For a sample game tree (left-to-right traversal):

```
       □ (ROOT/MAX)
      / \
     /   \
    ○     ○ (MIN)
   / \   / \
  □   □ □   □ (MAX)
  3   5 8   2 (Leaf values)
```

**Step-by-step alpha-beta values**:

1. **Start at root**: α=-∞, β=+∞
   
2. **Traverse left MIN node**:
   - α=-∞, β=+∞
   
   - **Examine left child (leaf=3)**:
     - Update α to 3 (best for MAX)
     - Node values: α=3, β=+∞
   
   - **Examine right child (leaf=5)**:
     - Update α to 5 (better for MAX)
     - Node values: α=5, β=+∞
   
   - **MIN node selects minimum**: 3
   - Propagate to parent: α=3, β=+∞

3. **Traverse right MIN node**:
   - α=3, β=+∞ (from parent)
   
   - **Examine left child (leaf=8)**:
     - Update α to 8 (best so far for MAX)
     - Node values: α=8, β=+∞
   
   - **Examine right child (leaf=2)**:
     - No change to α (8 is better than 2 for MAX)
     - Node values: α=8, β=+∞
   
   - **MIN node selects minimum**: 2
   - Since 2 < α (3), propagate to parent: α=3, β=2

4. **ROOT selects maximum** between children values (3 and 2): 3

**Final tree with alpha-beta values**:
```
       □ α=3, β=+∞
      / \
     /   \
    ○     ○ α=3, β=2
   / \   / \
  □   □ □   □ α=8, β=+∞
  3   5 8   2
```

Note: In a larger tree, when β ≤ α at any node, we can prune (skip examining) the remaining children of that node because they cannot influence the final decision.

### 6. [E] Given a policy, derive the reward function.

Deriving a reward function from a policy is a form of inverse reinforcement learning (IRL). Here's how to approach it:

**Process to derive a reward function from a policy**:

1. **Gather demonstrations**:
   - Observe the agent following the given policy π
   - Record state transitions and actions (s, a, s')
   - Create a dataset of trajectories

2. **Feature identification**:
   - Identify relevant features φ(s) for each state
   - These features should capture aspects that might influence decision-making

3. **Reward function parametrization**:
   - Assume a linear reward function: R(s) = w·φ(s) where w are weights
   - (Could also use non-linear functions like neural networks for complex behaviors)

4. **Maximum entropy IRL**:
   - The principle is that the demonstrated policy maximizes:
     `reward - entropy` or `Σ R(s) - α·log(π(a|s))`
   - Solve for weights w that make the demonstrated policy optimal

5. **Mathematical formulation**:
   - For deterministic policy in an MDP:
     `R(s) = min[Q(s,a') - Q(s,π(s))] for all a'≠π(s)`
   - Where Q values can be derived from the policy using policy evaluation

6. **Verification**:
   - Test if a new agent trained with the derived reward function learns a policy similar to the original

For example, if we observe a robot always taking the shortest path to a charging station when battery is low, we might derive a reward function with:
- Large negative rewards for low battery states
- Small negative rewards for movement (cost)
- Large positive rewards for reaching charging station

The derived reward function should make the observed policy optimal when solved using reinforcement learning algorithms.

### 7. [M] Pros and cons of on-policy vs. off-policy.

**On-Policy Methods**

*Pros*:
- **Stability**: More stable learning and convergence
- **Simplicity**: Conceptually simpler to understand and implement
- **Policy coherence**: What you learn is directly applicable to your current behavior
- **Sample efficiency during execution**: Less exploration means better performance while learning
- **Better for deterministic environments**: Can converge faster in simpler environments
- **Safety**: Better for scenarios where exploration could be costly or dangerous

*Cons*:
- **Sample inefficiency during learning**: Requires fresh samples for each policy update
- **Limited experience reuse**: Cannot efficiently use historical data from old policies
- **Exploration challenges**: Harder to explore thoroughly while maintaining policy coherence
- **Slower convergence** in complex environments requiring extensive exploration
- **Tendency toward local optima**: May get stuck in suboptimal solutions

**Off-Policy Methods**

*Pros*:
- **Experience reuse**: Can learn from demonstrations, historical data, or other agents
- **Sample efficiency**: Can reuse the same experiences for multiple updates
- **Data efficiency**: Better utilization of collected experience
- **Better exploration**: Can learn optimal policy while following exploratory policy
- **Flexibility**: Can learn multiple policies simultaneously from the same data stream
- **Offline learning**: Can learn entirely from pre-collected datasets

*Cons*:
- **Instability**: More prone to divergence and oscillations
- **Distribution mismatch**: Training distribution may differ from target policy distribution
- **Implementation complexity**: Typically more complex algorithms
- **Hyperparameter sensitivity**: Often require more careful tuning
- **Higher variance**: Importance sampling can increase variance in updates
- **Convergence issues**: May have theoretical convergence challenges in function approximation settings

**Examples**:
- **On-policy**: SARSA, PPO, TRPO, A2C
- **Off-policy**: Q-learning, DQN, DDPG, SAC, TD3

The choice depends on specific requirements: use on-policy for stability and safety, off-policy for data efficiency and when exploration is expensive.

### 8. [M] What's the difference between model-based and model-free? Which one is more data-efficient?

**Model-Based RL**

*Key characteristics*:
- Learns an explicit model of environment dynamics: P(s'|s,a) and R(s,a,s')
- Uses the model for planning and decision-making
- Can simulate experiences without interacting with environment
- Combines planning and learning

*Advantages*:
- **Superior data efficiency**: Requires fewer real environment interactions
- **Transfer learning**: Model can transfer to related tasks
- **Planning capability**: Can "look ahead" to evaluate future outcomes
- **Risk avoidance**: Can anticipate negative outcomes without experiencing them
- **Explicit uncertainty**: Can represent uncertainty about the environment

*Disadvantages*:
- **Model errors**: Performance limited by model accuracy
- **Computational cost**: Planning with models is computationally intensive
- **Complexity**: Harder to implement effectively
- **Scalability challenges**: Modeling complex environments is difficult

**Model-Free RL**

*Key characteristics*:
- Learns policy or value function directly from experience
- No explicit modeling of environment dynamics
- Relies on trial-and-error interactions
- Focus on "what to do" rather than "how the world works"

*Advantages*:
- **Simplicity**: Easier to implement and understand
- **Asymptotic performance**: Often achieves better final performance
- **Robustness**: No model bias or error accumulation
- **Scalability**: Can handle complex environments where modeling is difficult
- **Computational efficiency**: No planning overhead

*Disadvantages*:
- **Sample inefficiency**: Requires many environment interactions
- **Limited transfer**: Less transferable knowledge between tasks
- **Exploration challenges**: Difficulty with sparse rewards
- **Safety concerns**: Must experience failures to learn to avoid them

**Data Efficiency**: 
Model-based methods are significantly more data-efficient, often requiring orders of magnitude fewer interactions. This makes them preferable when:
- Real-world interactions are expensive or risky
- Sample collection is time-consuming
- Simulation is cheaper than real interaction
- Environment dynamics are relatively simple to model

Modern approaches often combine elements of both paradigms to get the best of both worlds.

## Other Deep Learning Topics

### 1. [M] An autoencoder is a neural network that learns to copy its input to its output. When would this be useful?

Autoencoders, despite their seemingly simple objective of reconstructing their input, have several valuable applications:

1. **Dimensionality reduction**:
   - Creating compact representations (embeddings) in the bottleneck layer
   - Alternative to PCA with capacity for non-linear relationships
   - Visualization of high-dimensional data

2. **Denoising**:
   - Train with noisy inputs but clean targets (denoising autoencoders)
   - Network learns to remove noise while preserving content
   - Applications in image restoration, audio enhancement

3. **Anomaly detection**:
   - Train on normal data, test on potentially anomalous samples
   - High reconstruction error indicates anomalies
   - Useful in fraud detection, manufacturing quality control, system monitoring

4. **Data compression**:
   - Neural compression systems for images, audio, video
   - Can outperform traditional codecs on specific data types
   - Learned codecs like in image compression

5. **Pre-training**:
   - Initialization for supervised learning tasks
   - Particularly useful with limited labeled data
   - Encoder can be used as feature extractor

6. **Generative modeling**:
   - Variational autoencoders (VAEs) learn probabilistic generative models
   - Generating new samples similar to training data
   - Controllable generation by manipulating latent space

7. **Missing value imputation**:
   - Reconstructing incomplete data
   - Learning the underlying data distribution to fill gaps

8. **Domain adaptation**:
   - Learning domain-invariant features
   - Transferring knowledge between related domains

9. **Information retrieval**:
   - Creating searchable embeddings of complex data types
   - Similarity search using latent space representations

The key insight is that the bottleneck forces the network to learn the most important features of the data, creating useful representations for downstream tasks.

### 2. Self-attention

#### [E] What's the motivation for self-attention?

Self-attention was motivated by several limitations in previous architectures:

1. **Capturing long-range dependencies**:
   - RNNs struggled with long sequences due to vanishing gradients
   - CNNs required deep networks to capture distant relationships
   - Self-attention directly connects any position with any other position

2. **Parallel computation**:
   - RNNs process sequences sequentially, limiting parallelization
   - Self-attention operates on the entire sequence simultaneously
   - Enables significantly faster training on modern hardware

3. **Context awareness**:
   - Each token/position needs awareness of its context
   - Self-attention explicitly models relationships between all positions
   - Allows dynamic, content-based interactions

4. **Interpretability**:
   - Attention weights provide insight into which parts of the input influence predictions
   - Creates explainable connections between elements

5. **Variable-length representations**:
   - Flexible handling of sequences of any length
   - No fixed receptive field limitations
   - Adaptable to different input types (text, images, audio)

6. **Inductive bias reduction**:
   - Minimal assumptions about data structure
   - Learns relationships based on content rather than position
   - Can discover patterns RNNs and CNNs might miss

These motivations led to the Transformer architecture, which has revolutionized NLP and is expanding into other domains like computer vision and reinforcement learning.

#### [E] Why would you choose a self-attention architecture over RNNs or CNNs?

You might choose self-attention architectures over RNNs or CNNs for these compelling reasons:

1. **Superior handling of long-range dependencies**:
   - Self-attention connects any position to any other position with constant computational path
   - RNNs have path length proportional to sequence distance
   - CNNs require many layers to connect distant elements

2. **Parallelization advantages**:
   - Self-attention processes all positions simultaneously
   - RNNs are inherently sequential
   - Result: drastically faster training on modern hardware

3. **Context modeling capability**:
   - Captures global context at every layer
   - Builds relationships based on content, not just proximity
   - Adaptive attention distribution based on relevance

4. **Architectural flexibility**:
   - No fixed receptive field constraints
   - Same architecture works for various sequence lengths
   - Easy scaling by adjusting number of layers/heads

5. **Better gradient flow**:
   - Direct connections prevent vanishing/exploding gradients
   - Stable training even for very deep networks

6. **State-of-the-art performance**:
   - Transformers have achieved SOTA results across NLP, CV, audio
   - Models like BERT, GPT, ViT demonstrate consistent superiority

7. **Interpretability benefits**:
   - Attention maps provide insights into model decisions
   - More transparent reasoning than CNN feature maps or RNN hidden states

8. **No assumptions about sequential order**:
   - Handles arbitrary sequences without inherent ordering bias
   - Beneficial for tasks with complex structural dependencies

The main trade-offs are higher memory requirements and computational costs for short sequences, where RNNs or CNNs might be more efficient.

#### [M] Why would you need multi-headed attention instead of just one head for attention?

Multi-headed attention provides several crucial advantages over single-headed attention:

1. **Parallel feature learning**:
   - Different heads can learn different aspects of relationships
   - One head might focus on syntactic relations, another on semantic similarities
   - Similar to how CNNs use multiple filters to detect different features

2. **Joint attention from different representation subspaces**:
   - Each head projects inputs into different subspaces
   - Allows attention to operate in multiple representation spaces simultaneously
   - Captures relationships that might be obscured in a single representation

3. **Increased model capacity**:
   - More parameters and representational power
   - Better modeling of complex dependencies
   - Higher ceiling on performance

4. **Ensemble-like effects**:
   - Multiple semi-independent attention mechanisms
   - Reduces variance in attention patterns
   - Provides robustness through diversified feature capture

5. **Specialized attention patterns**:
   - Some heads can focus on local context
   - Others can capture long-range dependencies
   - Enables both fine-grained and broad attention simultaneously

6. **Improved stability**:
   - Reduces risk of attention collapse (focusing on limited patterns)
   - More stable gradients during training

7. **Empirical performance gains**:
   - Consistent improvements observed in practice
   - Vaswani et al. showed performance increases with more heads (up to a point)

Example: In language translation, different heads might focus on different aspects:
- Word alignment between languages
- Syntactic structure
- Entity relationships
- Semantic similarity
- Topic coherence

The concatenation of these diverse attention patterns creates a richer representation than any single attention mechanism could provide.

#### [M] How would changing the number of heads in multi-headed attention affect the model's performance?

Changing the number of attention heads has several impacts on model performance:

**Increasing the number of heads**:

*Potential benefits*:
- **Expressivity**: More heads can capture more relationship types
- **Specialization**: Heads can become more focused on specific patterns
- **Robustness**: Ensemble-like effects reduce variance
- **Performance ceiling**: Higher potential upper bound on capability

*Potential drawbacks*:
- **Computational cost**: More heads = more computation
- **Memory usage**: Increases linearly with head count
- **Overfitting risk**: Too many heads may learn redundant or noise patterns
- **Diminishing returns**: Benefits plateau after a certain point
- **Training difficulty**: More parameters can make optimization harder

**Decreasing the number of heads**:

*Potential benefits*:
- **Efficiency**: Reduced computation and memory requirements
- **Easier optimization**: Fewer parameters to learn
- **Generalization**: May reduce overfitting on smaller datasets
- **Forced representation sharing**: May learn more robust features

*Potential drawbacks*:
- **Underfitting**: Insufficient capacity to model complex relationships
- **Information bottleneck**: Limited ability to capture diverse patterns
- **Performance ceiling**: Lower maximum capability

**Empirical findings**:
- Original Transformer paper used 8 heads
- Performance typically improves up to 8-16 heads for large models
- Some heads can be pruned with minimal performance impact (Michel et al., 2019)
- Recent work shows quality-efficiency trade-offs with variable numbers of heads per layer
- GPT models use different head counts at different layers (often more in middle layers)

The optimal head count depends on:
- Dataset size and complexity
- Model size and depth
- Computational constraints
- Task requirements

Finding the right balance often requires empirical tuning for your specific application.

### 3. Transfer learning

#### [E] You want to build a classifier to predict sentiment in tweets but you have very little labeled data (say 1000). What do you do?

With only 1,000 labeled tweets for sentiment analysis, transfer learning is your best approach:

**Step 1: Leverage pre-trained language models**
- Fine-tune models like BERT, RoBERTa, or DistilBERT (smaller, faster)
- These models already understand language structure from pre-training on massive corpora
- Twitter-specific models like BERTweet may be even better

**Step 2: Adapt to your domain**
- Implement gradual unfreezing - initially only train the output layer
- Use a low learning rate (1e-5 to 5e-5) to avoid catastrophic forgetting
- Consider intermediate domain-adaptive pre-training on unlabeled tweets before fine-tuning

**Step 3: Augment your limited data**
- Back-translation: Translate to another language and back to create paraphrases
- Synonym replacement: Replace non-sentiment words with synonyms
- Easy data augmentation (EDA): Random insertions, deletions, swaps
- Potentially use GPT-4 or similar to generate plausible variations of your labeled examples

**Step 4: Use semi-supervised learning**
- Self-training: Use your model to pseudo-label unlabeled tweets
- Consistency regularization: Ensure predictions are stable under small perturbations
- Unlabeled tweet collection is typically easy and abundant

**Step 5: Optimize for small data**
- Strong regularization (dropout, weight decay)
- Early stopping based on validation performance
- Ensemble multiple fine-tuned models with different initializations
- Consider simpler models (Linear SVM on top of frozen embeddings) as baselines

This approach typically achieves 85-90%+ accuracy on sentiment analysis even with limited labeled data, significantly outperforming training from scratch.

#### [M] What's gradual unfreezing? How might it help with transfer learning?

**Gradual unfreezing** is a transfer learning technique where you progressively unfreeze layers of a pre-trained model during fine-tuning, starting from the output layer and moving toward the input layer.

**How it works**:

1. **Initial state**: All layers except the output/task-specific layer(s) are frozen (weights not updated)

2. **Progressive unfreezing**:
   - First train only the new output layer(s) for a few epochs
   - Then unfreeze the last pre-trained layer and train both together
   - Continue progressively unfreezing earlier layers
   - Typically unfreeze one layer at a time after 1-2 epochs

3. **Layer-specific learning rates**:
   - Often combined with discriminative learning rates
   - Lower learning rates for earlier layers (closer to input)
   - Higher learning rates for later layers (closer to output)

**Benefits for transfer learning**:

1. **Prevents catastrophic forgetting**:
   - Preserves useful general features learned during pre-training
   - Avoids destroying valuable representations in early layers

2. **Hierarchical adaptation**:
   - Respects the hierarchical nature of neural networks
   - Lower layers capture general features that transfer well
   - Higher layers capture task-specific features that need more adaptation

3. **Better generalization**:
   - Reduces overfitting on small target datasets
   - Maintains regularizing effect of pre-training

4. **Computational efficiency**:
   - Initial epochs are faster (fewer parameters to update)
   - Allows early assessment of transfer potential

5. **Systematic fine-tuning**:
   - More controlled adaptation process
   - Easier to determine optimal unfreezing schedule

This technique has proven particularly effective for NLP tasks with pre-trained language models (ULMFiT, BERT) and computer vision tasks with pre-trained convolutional networks, especially when the target dataset is small relative to the model size.

### 4. Bayesian methods

#### [M] How do Bayesian methods differ from the mainstream deep learning approach?

**Bayesian Deep Learning vs. Mainstream Deep Learning**

*Key philosophical differences*:
- **Mainstream**: Seeks point estimates of parameters through optimization
- **Bayesian**: Learns entire probability distributions over parameters

*Core technical differences*:

1. **Uncertainty representation**:
   - **Mainstream**: Usually only captures aleatoric uncertainty (data noise)
   - **Bayesian**: Explicitly models both epistemic uncertainty (model uncertainty) and aleatoric uncertainty

2. **Learning approach**:
   - **Mainstream**: Maximum likelihood estimation or MAP estimation
   - **Bayesian**: Posterior inference (full Bayesian) or approximations (variational inference, MCMC)

3. **Regularization mechanism**:
   - **Mainstream**: Explicit regularization (dropout, weight decay, data augmentation)
   - **Bayesian**: Natural regularization from prior distributions and model averaging

4. **Output format**:
   - **Mainstream**: Single prediction (sometimes with confidence score)
   - **Bayesian**: Predictive distribution capturing uncertainty

5. **Training process**:
   - **Mainstream**: Optimization of loss function (typically via SGD variants)
   - **Bayesian**: Approximating or sampling from posterior distribution

6. **Parameter handling**:
   - **Mainstream**: Learned as fixed values
   - **Bayesian**: Treated as random variables with distributions

7. **Knowledge incorporation**:
   - **Mainstream**: Primarily through architecture design
   - **Bayesian**: Explicitly through prior distributions plus architecture

8. **Computational approach**:
   - **Mainstream**: Typically deterministic forward passes
   - **Bayesian**: Often involves sampling or ensemble-like approaches

Bayesian deep learning approaches include Bayes by Backprop, Monte Carlo Dropout, deep ensembles (approximate Bayesian), and variational autoencoders (for generative models).

#### [M] How are the pros and cons of Bayesian neural networks compared to the mainstream neural networks?

**Bayesian Neural Networks**

*Pros*:
- **Uncertainty quantification**: Explicitly models confidence in predictions
- **Overfitting resistance**: Natural regularization from parameter averaging
- **Out-of-distribution detection**: Better identifies when inputs are unlike training data
- **Sample efficiency**: Often performs better with limited data
- **Principled ensembling**: Automatic model averaging through posterior sampling
- **Active learning compatibility**: Can guide data collection by targeting high-uncertainty regions
- **Prior knowledge integration**: Directly incorporate domain expertise via priors
- **Calibrated probabilities**: Output probabilities tend to better reflect true likelihoods

*Cons*:
- **Computational cost**: Typically 2-10× more expensive than standard NNs
- **Implementation complexity**: More difficult to implement and debug
- **Increased inference time**: Multiple forward passes or sampling often required
- **Scaling challenges**: Harder to scale to very large models
- **Prior specification difficulty**: Choosing appropriate priors can be non-trivial
- **Approximate inference**: True Bayesian inference is usually intractable, requiring approximations
- **Limited software support**: Fewer optimized frameworks compared to standard deep learning
- **Training instability**: Some methods have convergence challenges

**Mainstream Neural Networks**

*Pros*:
- **Computational efficiency**: Faster training and inference
- **Implementation simplicity**: Well-established frameworks and techniques
- **Scalability**: Proven track record with billion-parameter models
- **Optimization focus**: Highly optimized algorithms for point estimation
- **Performance on large datasets**: State-of-the-art results when data is abundant
- **Hardware optimization**: Better supported by specialized hardware (GPUs, TPUs)
- **Ecosystem maturity**: Extensive tooling, libraries, and community support

*Cons*:
- **Poor uncertainty estimation**: Overconfident predictions, especially out-of-distribution
- **Overfitting tendency**: Requires explicit regularization techniques
- **Data inefficiency**: Generally needs more data to generalize well
- **Calibration issues**: Output probabilities often poorly calibrated
- **Ensemble overhead**: Requires explicit ensemble implementation
- **Knowledge integration challenges**: Harder to incorporate prior knowledge
- **Point estimates**: Ignores parameter uncertainty

Hybrid approaches like Monte Carlo Dropout allow mainstream architectures to gain some Bayesian benefits with limited additional cost.

#### [M] Why do we say that Bayesian neural networks are natural ensembles?

Bayesian neural networks (BNNs) are considered natural ensembles for several fundamental reasons:

1. **Multiple model integration**:
   - BNNs effectively integrate predictions from infinitely many neural networks
   - Each sample from the posterior represents a different model configuration
   - Predictions are averaged over multiple parameter configurations

2. **Parameter distribution vs. point estimates**:
   - Standard NNs learn single parameter values (w*)
   - BNNs learn distributions over parameters p(w|D)
   - Prediction involves integrating over this distribution:
     p(y|x,D) = ∫ p(y|x,w) p(w|D) dw

3. **Automatic weighting mechanism**:
   - Models more consistent with data get higher posterior probability
   - Naturally weights models by their likelihood × prior
   - No need for manual ensemble weighting schemes

4. **Diverse model capture**:
   - Posterior captures multiple good solutions in parameter space
   - Especially important when multiple distinct solutions exist
   - Handles multimodal posterior distributions

5. **Theoretical foundation**:
   - Bayesian model averaging is theoretically optimal under the model's assumptions
   - Minimizes expected predictive error
   - Properly accounts for parameter uncertainty

6. **Practical implementation**:
   - Monte Carlo methods approximate integration by sampling
   - y_pred ≈ (1/M) Σ f(x; w_m) where w_m ~ p(w|D)
   - Each forward pass with different weights resembles a different ensemble member

This ensemble behavior gives BNNs their characteristic benefits: better uncertainty estimates, improved generalization, and robustness to overfitting—similar to traditional ensembles but arising naturally from Bayesian principles rather than explicit construction.

### 5. GANs

#### [E] What do GANs converge to?

In theory, GANs converge to a Nash equilibrium where:

1. **The generator** produces a data distribution perfectly matching the real data distribution:
   - p_g(x) = p_data(x)
   - The generated samples become indistinguishable from real data

2. **The discriminator** reaches a state of maximum confusion:
   - D(x) = 0.5 for all inputs
   - Unable to distinguish between real and generated samples
   - Gives 50% probability to both real and fake samples

More formally, GANs aim to minimize the Jensen-Shannon divergence (JSD) between the generated and real data distributions. At convergence:
- JSD(p_data || p_g) = 0
- This occurs only when p_data = p_g

In practice, perfect convergence rarely happens due to:
- **Non-convexity**: The optimization landscape has many local minima
- **Mode collapse**: Generator might focus on a subset of the real data distribution
- **Training instability**: Oscillations between generator and discriminator
- **Finite capacity**: Models have limited capacity to represent distributions
- **Finite data**: Training data provides only an approximate view of the true distribution

Instead, practical GAN training often reaches an approximate equilibrium where the generator produces convincing but not perfect samples, and the discriminator maintains some ability to detect subtle differences between real and generated data.

#### [M] Why are GANs so hard to train?

GANs are notoriously difficult to train for several interconnected reasons:

1. **Unstable dynamics**:
   - Two networks constantly adapting to each other creates moving targets
   - Minimax game creates oscillations and spiraling behavior
   - Small changes in one network can dramatically affect the other's landscape

2. **Vanishing gradients**:
   - If discriminator becomes too good, generator gradients vanish
   - Generator receives no useful signal for improvement
   - Early GAN formulations especially vulnerable to this

3. **Mode collapse**:
   - Generator maps different inputs to same output modes
   - Produces limited variety of samples
   - Ignores portions of the target distribution

4. **Balance problems**:
   - Training pace mismatch between generator and discriminator
   - If either network learns too quickly, the system fails
   - Requires careful synchronization of learning rates

5. **Convergence issues**:
   - No clear convergence criteria or guarantees
   - Nash equilibrium hard to reach in high-dimensional non-convex space
   - Difficult to know when to stop training

6. **Evaluation challenges**:
   - No single objective metric indicates success
   - Visual inspection often needed but subjective
   - Metrics like Inception Score or FID are proxies with limitations

7. **Sensitivity to hyperparameters**:
   - Extremely sensitive to architecture choices
   - Learning rates, batch size, and normalization critically important
   - Small changes can mean difference between success and failure

8. **Gradient problems**:
   - Non-saturating loss functions needed to prevent gradient issues
   - Careful initialization required
   - Gradient penalties often necessary (as in WGAN-GP)

Modern techniques addressing these issues include:
- Wasserstein distance (WGAN)
- Spectral normalization
- Two-timescale update rule (TTUR)
- Progressive growing
- Self-attention mechanisms
- Consistency regularization
- Advanced architectures (StyleGAN, BigGAN)

These innovations have made GANs more stable but still more challenging to train than standard supervised models.

## Training Neural Networks

### 1. [E] When building a neural network, should you overfit or underfit it first?

When building a neural network, you should **aim to overfit first**, then apply regularization to prevent overfitting. This approach follows a successful model development workflow:

1. **Start by overfitting**:
   - Verify that your model has sufficient capacity to learn the task
   - Ensure your optimization process works correctly
   - Confirm that your data pipeline and loss function are properly implemented
   - Initially use little or no regularization
   - Train until you achieve very low training error

2. **Signs of successful overfitting**:
   - Training loss approaches zero
   - Training accuracy approaches 100%
   - Validation performance significantly worse than training

3. **Why this approach works**:
   - If you can't overfit, you likely have fundamental problems:
     - Model may lack capacity
     - Optimization might be failing
     - Data might have issues (incorrectly labeled, too noisy)
     - Architectural design might be inappropriate for the task

4. **After achieving overfitting**:
   - Gradually introduce regularization (dropout, weight decay, data augmentation)
   - Tune hyperparameters to improve validation performance
   - Add early stopping based on validation metrics
   - Consider reducing model complexity if needed

Andrew Ng summarizes this approach as: "If your training error is not low, you have a high bias problem. If your training error is low but validation error is high, you have a high variance problem." Overfitting first ensures you've solved the bias problem before addressing variance.

### 2. [E] Write the vanilla gradient update.

The vanilla gradient update (also known as gradient descent) is:

```
θ = θ - η * ∇J(θ)
```

Where:
- θ (theta) represents the model parameters (weights and biases)
- η (eta) is the learning rate, controlling step size
- ∇J(θ) is the gradient of the cost function J with respect to parameters θ

For a specific parameter w:
```
w = w - η * ∂J/∂w
```

For a neural network with multiple layers, this is applied to each parameter individually.

In practice, this update is typically performed in batches (mini-batch gradient descent):

```
θ = θ - η * ∇J(θ; X_batch, y_batch)
```

The vanilla gradient update serves as the foundation for more advanced optimization algorithms like momentum, RMSprop, Adam, etc., which build on this basic update rule by adding various modifications to improve convergence speed and stability.

### 3. Neural network in simple Numpy

#### [E] Write in plain NumPy the forward and backward pass for a two-layer feed-forward neural network with a ReLU layer in between.

```python
import numpy as np

# Network architecture: input -> linear -> ReLU -> linear -> output

def forward_pass(X, W1, b1, W2, b2):
    # First layer (linear)
    Z1 = np.dot(X, W1) + b1
    
    # ReLU activation
    A1 = np.maximum(0, Z1)
    
    # Second layer (linear)
    Z2 = np.dot(A1, W2) + b2
    
    # Cache values for backward pass
    cache = (X, Z1, A1, W1, W2)
    
    return Z2, cache

def backward_pass(dZ2, cache, learning_rate):
    # Unpack cached values
    X, Z1, A1, W1, W2 = cache
    m = X.shape[0]  # Batch size
    
    # Gradients for second layer
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, W2.T)
    
    # Gradient for ReLU activation
    dZ1 = dA1.copy()
    dZ1[Z1 <= 0] = 0  # ReLU gradient: 1 if Z1 > 0, 0 otherwise
    
    # Gradients for first layer
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    
    # Update parameters
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    b1 -= learning_rate * db1
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2

# Example usage:
# For binary classification with MSE loss:
def train_step(X, y, W1, b1, W2, b2, learning_rate):
    # Forward pass
    y_pred, cache = forward_pass(X, W1, b1, W2, b2)
    
    # Compute error (MSE loss gradient is simply prediction - target)
    dZ2 = y_pred - y
    
    # Backward pass
    W1, b1, W2, b2 = backward_pass(dZ2, cache, learning_rate)
    
    return W1, b1, W2, b2, np.mean((y_pred - y)**2)  # Return loss too
```

This implementation showcases:
1. Clean separation of forward and backward passes
2. Proper handling of the ReLU activation and its gradient
3. Parameter updates using vanilla gradient descent
4. Caching intermediate values for efficient backpropagation
5. Vectorized operations for efficiency

For a complete implementation, you'd add initialization, full training loop, and prediction functions.

#### [M] Implement vanilla dropout for the forward and backward pass in NumPy.

```python
import numpy as np

def forward_with_dropout(X, W1, b1, W2, b2, keep_prob=0.8, is_training=True):
    """
    Forward pass with dropout regularization
    
    Parameters:
    - keep_prob: probability of keeping a neuron active
    - is_training: flag for training/inference mode
    """
    # First layer (linear)
    Z1 = np.dot(X, W1) + b1
    
    # ReLU activation
    A1 = np.maximum(0, Z1)
    
    # Dropout mask (only during training)
    if is_training:
        D1 = np.random.rand(*A1.shape) < keep_prob
        A1 = np.multiply(A1, D1)  # Apply mask
        A1 = A1 / keep_prob  # Scale to maintain expected value
    else:
        D1 = None  # No dropout during inference
    
    # Second layer (linear)
    Z2 = np.dot(A1, W2) + b2
    
    # Cache values for backward pass
    cache = (X, Z1, A1, D1, W1, W2, keep_prob)
    
    return Z2, cache

def backward_with_dropout(dZ2, cache, learning_rate):
    """
    Backward pass with dropout
    """
    # Unpack cached values
    X, Z1, A1, D1, W1, W2, keep_prob = cache
    m = X.shape[0]  # Batch size
    
    # Gradients for second layer
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, W2.T)
    
    # Apply dropout mask to gradients
    if D1 is not None:  # Only during training
        dA1 = np.multiply(dA1, D1)
        dA1 = dA1 / keep_prob  # Scale gradients
    
    # Gradient for ReLU activation
    dZ1 = dA1.copy()
    dZ1[Z1 <= 0] = 0
    
    # Gradients for first layer
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    
    # Update parameters
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    b1 -= learning_rate * db1
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2

# Example usage
def train_step_with_dropout(X, y, W1, b1, W2, b2, learning_rate, keep_prob=0.8):
    # Forward with dropout (training mode)
    y_pred, cache = forward_with_dropout(X, W1, b1, W2, b2, keep_prob, is_training=True)
    
    # Compute loss gradient
    dZ2 = y_pred - y
    
    # Backward with dropout
    W1, b1, W2, b2 = backward_with_dropout(dZ2, cache, learning_rate)
    
    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2, keep_prob=0.8):
    # Forward without dropout (inference mode)
    y_pred, _ = forward_with_dropout(X, W1, b1, W2, b2, keep_prob, is_training=False)
    return y_pred
```

Key dropout implementation details:
1. **Generation of binary mask** `D1` with probability `keep_prob` of keeping each neuron
2. **Scaling activation values** by `1/keep_prob` during training to maintain expected output
3. **Applying dropout only during training** (controlled by `is_training` flag)
4. **Reusing the same dropout mask** during backpropagation
5. **Scaling gradients** in the same way as activations
6. **No dropout during inference** for deterministic predictions

This maintains the expected values of activations and properly propagates gradients through the dropped-out neurons while providing regularization benefits.

### 4. Activation functions

#### [E] Draw the graphs for sigmoid, tanh, ReLU, and leaky ReLU.

While I can't physically draw graphs, here are the mathematical descriptions of these activation functions:

**Sigmoid: σ(x) = 1 / (1 + e^(-x))**
- Range: (0, 1)
- S-shaped curve
- Approaches 0 as x → -∞
- Approaches 1 as x → +∞
- Centered at (0, 0.5)

**Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))**
- Range: (-1, 1)
- S-shaped curve
- Approaches -1 as x → -∞
- Approaches 1 as x → +∞
- Centered at (0, 0)

**ReLU: f(x) = max(0, x)**
- Range: [0, ∞)
- Linear for x > 0
- Zero for x ≤ 0
- Sharp bend at origin (0, 0)

**Leaky ReLU: f(x) = max(αx, x) where α is a small constant (e.g., 0.01)**
- Range: (-∞, ∞)
- Linear with slope 1 for x > 0
- Linear with small slope α for x < 0
- Slight bend at origin (0, 0)

The key visual distinctions:
- Sigmoid and tanh are smooth, bounded functions
- ReLU has a "hinge" shape with flat region for negative values
- Leaky ReLU looks similar to ReLU but has a slight slope in the negative region

#### [E] Pros and cons of each activation function.

**Sigmoid (Logistic Function)**

*Pros*:
- Smooth gradient, preventing "jumps" in output values
- Output bound between 0 and 1, good for probability interpretation
- Clear prediction in binary classification (>0.5 vs <0.5)
- Historically important, well-understood mathematics

*Cons*:
- Suffers from vanishing gradient problem for very positive/negative inputs
- Outputs not zero-centered, causing zig-zag dynamics in gradient descent
- Computationally expensive (involves exponential)
- Network training typically slower than with ReLU variants
- Prone to saturation - "kills" gradients when saturated

**Tanh (Hyperbolic Tangent)**

*Pros*:
- Zero-centered outputs