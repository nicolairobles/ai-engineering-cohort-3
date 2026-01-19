<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# LLM Foundations - Week 1 Summary

## Course Overview

- **Topic**: LLM (Large Language Model) Foundations
- **Focus**: Understanding how LLMs are built, trained, and work in practice
- **Importance**: Advanced applications (agents, reasoning models, code platforms, productivity tools) are built on top of LLMs


## Course Structure

- Overview of LLMs
- Two main training stages:
    - Pretraining
    - Post-training
- Hands-on project: Build an LLM playground in Python

***

## What is an LLM?

- **Definition**: AI model that can understand and generate text
- **Capabilities**:
    - Answer questions
    - Engage in conversations
    - Respond to follow-up questions


### Popular LLM Chatbots

- **ChatGPT** (OpenAI) - First publicly available chatbot
- **Claude** (Anthropic)
- **Gemini** (Google)
- **Grok** (xAI)
- **Meta AI** (Meta)


### Common Features

- Simple UI with text input
- Model selection options
- Additional tools (agents, deep research, image generation, web search)
- Different response styles, tones, and detail levels across models

***

## Two Main Stages of Training LLMs

### Stage 1: Pretraining

- **Process**: Train model on internet data using training algorithms
- **Resources Required**:
    - Thousands of GPUs
    - Months of training
    - Hundreds of millions of dollars
- **Output**: Base model with implicit knowledge of the world
- **Who Can Do It**: Only well-funded startups and large companies
- **Example Costs** (from Stanford report):
    - GPT-3 (175B parameters): <\$10M
    - GPT-4 and Gemini Ultra: Hundreds of millions of dollars


### Stage 2: Post-training

- **Process**: Continue training base model on post-training data
- **Resources Required**:
    - Hundreds of GPUs (or fewer)
    - Days of training
    - Significantly less expensive than pretraining
- **Output**: Final model deployed for chatbot services

***

## Data Preparation for Pretraining

### Step 1: Crawling the Internet

- **Definition**: Software that extracts content and follows links across the web
- **Process**:

1. Start from base URL(s)
2. Extract content
3. Identify outgoing links
4. Visit those links
5. Repeat until majority of internet is visited


#### Web Crawler Components

- **Base URL**: Starting point
- **Visited set**: Tracks already-explored URLs
- **To-visit list**: Queue of discovered but unexplored URLs
- **Main loop**: Continuously explores unseen URLs
- **Libraries**: requests (HTTP requests), Beautiful Soup (HTML parsing)


#### Two Crawling Approaches

1. **Self-crawling**:
    - Used by big companies (OpenAI, Anthropic)
    - Provides more flexibility
2. **Public repositories**:
    - Example: Common Crawl (nonprofit organization)
    - **Statistics**:
        - Operating since 2008
        - ~2.7 billion web pages per crawl
        - 200-400 terabytes of HTML text per crawl
        - New crawl released monthly or bi-monthly
    - Preferred by startups and smaller companies for faster iteration

### Step 2: Data Cleaning

- **Purpose**: Extract useful information from raw HTML content


#### Issues to Address

- **HTML tags and markup**: Remove irrelevant tags, attributes, and formatting
- **Content extraction**: Focus on actual text content (headings, paragraphs) visible in browsers
- **Duplicate content**: Remove repeated text across websites to prevent memorization
- **Quality control**: Filter out unsafe or non-useful content from certain websites


#### Process

- Clean raw HTML content
- Extract clean text
- Name and organize the cleaned dataset

***

## Key Takeaways

- LLMs require two-stage training (pretraining + post-training)
- Pretraining is extremely expensive and resource-intensive
- Data preparation is critical and involves crawling and extensive cleaning
- Only major companies can afford full pretraining from scratch
- Different LLMs respond differently to the same prompts based on their training

## Part 2

<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Cleaned Datasets \& Tokenization Summary

## Popular Cleaned Datasets

### C4 (Colossal Clean Crawled Corpus)

- **Created by**: Google
- **Published on**: TensorFlow Datasets (tensorflow.org)
- **Description**: Clean version of Common Crawl web crawls
- **Structure**:
    - Table format with two columns:
        - Text column (extracted content)
        - URL column (source of the text)
- **Size variations**:
    - English data: ~305 GB
    - Non-clean version: ~2.3 TB
- **Importance**: Widely used in early days of LLM pretraining as a starting point
- **Note**: AI2 published a "cleaned version of C4" (further refinement)


### The Pile

- **Scale**: 3 trillion tokens (words/subwords)
- **Description**: Open corpus for language model pretraining research
- **Data sources** (beyond Common Crawl):
    - GitHub (code repositories)
    - Reddit
    - Wikipedia
    - Other specialized sources
- **Pipeline includes**:
    - Quality filtering
    - Content filtering
    - Multiple data source integration


### RefinedWeb

- **Characteristics**: Another popular cleaned dataset
- **Process**: Sequence of filtering and cleaning steps applied to raw internet data or Common Crawl
- **Available with detailed statistics on data size and composition**

***

## FineWeb Dataset \& Pipeline

### Overview

- **Publisher**: Hugging Face
- **Status**: Openly available
- **Type**: Comprehensive data cleaning pipeline with full documentation


### Cleaning Pipeline Steps

#### 1. URL Filtering

- **Purpose**: Remove unwanted content categories
- **Method**: Blocklist-based filtering
- **Example categories excluded**:
    - Adult content
    - Other subjective categories (varies by company preference)


#### 2. Text Extraction

- **Purpose**: Extract actual readable text from raw HTML
- **Focus**: Content within specific HTML tags (h1, p, etc.)
- **Removes**: HTML tags, markdown, attributes, irrelevant formatting


#### 3. Language Filtering

- **Purpose**: Filter text from languages the LLM shouldn't support
- **Customizable**: Based on target language requirements


#### 4. Deduplication

- **Purpose**: Remove very similar or duplicate content
- **Rationale**: Prevent LLM from memorizing repeated information


#### 5. Additional Quality Filters

- Various filtering steps for content quality and safety


#### 6. PII (Personally Identifiable Information) Removal

- **Purpose**: Remove sensitive information
- **Examples removed**:
    - Bank account numbers
    - Phone numbers
    - Other personal identifiers


### FineWeb Output Statistics

- **Storage size**: 44 terabytes of disk space
- **Total tokens**: ~15 trillion tokens (words/subwords)
- **Format**:
    - Large text file(s) or multiple text files
    - Table structure with text and URL columns
- **Content diversity**: Random internet text covering huge range of topics, domains, and areas
    - Example snippets: "Did you know you have to be the yellow nine volt..."
    - "Five reasons I love Boston..."

***

## Step 3: Tokenization

### Purpose

- **Goal**: Convert clean text data into long sequences of discrete numbers
- **Rationale**: Machine learning models require numerical inputs, not text strings


### Key Concept

- Tokenization is the bridge between human-readable text and machine-readable numbers

***

## How Tokenization Works

### Two Main Phases

#### Phase 1: Training Phase (One-time Setup)

**Input**: Long sequence of cleaned text data

**Step 1: Text Splitting**

- **Process**: Apply logic/algorithm to split text into smaller units (tokens)
- **Example**:
    - Input: "Machine learning (ML) is a subfield of artificial intelligence"
    - Output: ["machine", "learning", "ML", "is", "a", "sub", "field", "artificial", "intelligence", "."]
- **Result**: Very large list of smaller text chunks from entire cleaned internet data

**Step 2: Building the Vocabulary**

- **Process**:

1. Find all unique tokens from the split text list
2. Assign unique ID to each token
3. Create mapping table
- **Example vocabulary**:

```
ID    | Token
------|--------
0     | a
1     | about
2     | after
3     | all
4     | also
...   | ...
200,131 | [last token]
```

- **Size**: Vocabulary size depends on unique tokens in training data (example: ~200,131 tokens)
- **Storage**: Tokenizer internally stores this vocabulary


#### Phase 2: Inference Phase (Encoding \& Decoding)

**Encoding: Text → Numbers**

1. **Text Splitting**: Apply same splitting logic to input text
    - Example: "tell me a joke" → ["tell", "me", "a", "joke"]
2. **ID Replacement**: Look up each token in vocabulary and replace with corresponding ID
    - Example: ["tell", "me", "a", "joke"] →

**Decoding: Numbers → Text**

1. **Inverse Lookup**: Use inverse vocabulary table (ID → Token mapping)
2. **Token Retrieval**: Find corresponding token for each ID
3. **Text Reconstruction**: Apply inverse of text splitting to reconstruct original text
    - Example:  → ["tell", "me", "a", "joke"] → "tell me a joke"

***

## Summary of Data Preparation

### Three Main Steps

1. **Crawling**: Collect data from internet (self-crawl or use Common Crawl)
2. **Cleaning**: Apply filtering pipeline to get clean text (or use open-source datasets like FineWeb)
3. **Tokenization**: Convert clean text to numerical sequences

### Key Takeaways

- **Don't need to build from scratch**: Can start with open-source cleaned datasets (C4, FineWeb, The Pile, RefinedWeb)
- **Tokenization is bidirectional**: Can convert text→numbers (encoding) and numbers→text (decoding)
- **Vocabulary is fixed**: Created once during training phase, then used repeatedly during inference
- **Result**: Long sequence of discrete numbers ready for machine learning model training

## Part 3
<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Tokenizer Categories \& Algorithms Summary

## Three Categories of Tokenizers

### Key Context

- Different tokenization algorithms primarily differ in their **text splitting logic**
- Building vocabulary is straightforward: find all unique tokens and assign IDs
- **Word-level and character-level tokenizers are no longer used** for advanced LLMs
- **Subword-level tokenizers** are the current standard for modern LLMs

***

## 1. Word-Level Tokenizers

### How It Works

- **Splitting logic**: Split text based on white spaces
- **Example**:
    - Input: "perfectly fine"
    - Tokens: ["perfectly", "fine"]
    - Output IDs: [52,XXX, X,XXX] (numbers assigned from vocabulary)


### Characteristics

- **Vocabulary size**: Very large (hundreds of thousands of tokens)
- **Reason for size**: Countless unique words exist on the internet across languages
- **Token examples**: Individual complete words like "perfectly", "fine", "walking"


### Limitations

- **Huge vocabulary = expensive training**
    - Must maintain and manage hundreds of thousands of tokens
    - Model must learn associations with each token/ID
    - Very computationally expensive during training
- **Inefficiency**: Includes rare/meaningless words that appeared only a few times
    - Example: Random character sequences, unusual names, typos become individual tokens
    - Not efficient to give each rare word its own token


### Example Vocabulary

- **Size**: ~207,000+ tokens
- **Contents**: Every individual word seen on the internet
- Includes random/rare sequences that may not be meaningful

***

## 2. Character-Level Tokenizers

### How It Works

- **Splitting logic**: Split text based on individual characters
- **Example**:
    - Input: "perfectly fine"
    - Tokens: ["p", "e", "r", "f", "e", "c", "t", "l", "y", " ", "f", "i", "n", "e"]
    - Output IDs: [small numbers for each character]


### Characteristics

- **Vocabulary size**: Very small (~105 tokens in example)
- **Contents**:
    - Lowercase letters (a-z)
    - Uppercase letters (A-Z)
    - Punctuation marks (., !, ?, etc.)
    - Spaces and special characters
- **IDs**: Small numbers (since vocabulary is small)


### Limitations

- **Long sequences = expensive training**
    - Converting text to characters creates very long number sequences
    - Example: "perfectly fine" becomes 14+ tokens
    - Model must learn dependencies and relations between many tokens
    - Computationally costly to process long sequences

***

## 3. Subword-Level Tokenizers ✓ (Current Standard)

### Key Intuition

- **Goal**: Balance between word-level and character-level
- **Token size**: Larger than characters, smaller than words
- Achieves compromise: moderate vocabulary size + reasonable sequence length


### How It Works

- **Example**:
    - Input: "perfectly fine"
    - Tokens: ["perfect", "ly", "fine"]
    - Each token gets replaced by its ID


### Characteristics

- **Vocabulary size**: 50,000 - 200,000 tokens (controllable)
- **Token types**:
    - **Frequent complete words**: "the", "of", "home"
    - **Common subword units**: "ing", "ed", "able", "ly"
    - **Space + word combinations**: " love", " machine"
- **Flexibility**: Can handle rare/unknown words by breaking into subword units


### Example Vocabulary (~50,000 tokens)

- Complete words: "the", "of", "home", "perfectly", "walking"
- Subword units: "ing", "ed", "able", "ly"
- Allows splitting: "walking" → ["walk", "ing"]

***

## Byte Pair Encoding (BPE) Algorithm

### Overview

- **Most popular subword tokenization algorithm**
- Used by majority of modern LLMs (GPT-2, GPT-3, LLaMA 3, etc.)
- Provides full control over vocabulary size


### How BPE Works

#### Training Process

1. **Start with characters**: Create token for each unique character
2. **Count frequencies**: Track how often each word appears in training data
3. **Iterative merging**:
    - Find most frequent pair of adjacent tokens
    - Merge them into a new token
    - Add new token to vocabulary
    - Repeat until desired vocabulary size reached

#### Example Visualization

- Training data contains words with frequencies:
    - "hug" appears 10 times
    - Other words with "u" + "g" appear multiple times
- **Merge decision**: "u" + "g" → "ug" (new token)
    - Frequency: "ug" appears in "hug" (10×) + other words (5×) + more (5×) = very frequent
    - Algorithm creates "ug" as standalone token
- **Continues merging** until vocabulary reaches target size (e.g., 50,000 tokens)


### LLM Examples Using BPE

#### GPT-2 (OpenAI)

- **Section**: "Input Representation" in paper
- **Tokenizer**: BPE with variations
- Explicitly discusses using Byte Pair Encoding


#### LLaMA 3 (Meta)

- **Relatively new model**
- **Tokenizer**: "BPE model" (stated in paper)
- Continues the BPE standard


### Learning Resource

- **Hugging Face Tutorial**: "Byte Pair Encoding Tokenization"
- Detailed explanation of training algorithm
- Shows how tokens are formed and merged step-by-step

***

## Vocabulary Comparison

### Character-Level Example

- **Size**: 105 tokens
- **Contents**: a, b, c...z, A, B, C...Z, punctuation, space, numbers
- **Very small** but creates very long sequences


### Word-Level Example

- **Size**: 207,000+ tokens
- **Contents**: Every unique word from internet
- **Includes**: Rare words, misspellings, random character sequences, unusual names
- **Problem**: Inefficient—gives rare/meaningless words their own tokens


### Subword-Level Example (BPE)

- **Size**: 50,000 - 200,000 tokens (modern LLMs trend toward 100K-200K)
- **Contents**:
    - Frequent complete words: "the", "of", "home", "perfectly"
    - Common subwords: "ing", "ed", "able", "ly"
    - Hybrid units: " love", " machine" (space + word)
- **Balanced**: Efficient vocabulary + reasonable sequence lengths

***

## Tiktokenizer Visualization Tool

### Website

- **URL**: tiktokenizer (visualization tool)
- **Purpose**: Understand how text is tokenized and view IDs


### Features

- **Pre-trained tokenizers**: Choose from popular LLM tokenizers (already trained on internet data with internal vocabularies)
- **Real-time visualization**: See how text splits into tokens
- **ID display**: View the specific ID assigned to each token


### Example Usage

#### Example 1: "I love machine learning!"

- **Token count**: 5 tokens
- **Tokens**: ["I", " love", " machine", " learning", "!"]
- **IDs**: [40, 3021, ..., 0]
- **Note**: " love" (space + love) is a single token because BPE algorithm found it frequent in training data


#### Example 2: "I love walking"

- **Result**: "walking" = single token (frequent word in vocabulary)


#### Example 3: "It's perfectly fine"

- **"perfectly" with typo**: Splits into 3 tokens (not in vocabulary as misspelled)
- **"perfectly" correctly spelled**: Single token (complete word in vocabulary)
- **"ly"**: Has its own token (common English suffix)


#### Example 4: Random/rare text

- **Input**: Random character sequences or very rare words
- **Result**: Splits into many smaller subword units
- **Reason**: No tokens exist for these rare/unknown words in vocabulary
- **BPE handles gracefully**: Falls back to smaller units (characters if needed)


### Interactive Exploration

- Remove spaces to see tokenization changes
- Test common vs. rare words
- Observe how misspellings affect tokenization
- Understand why certain units become tokens (frequency-based)

***

## Key Takeaways

1. **Subword tokenization (especially BPE) is the modern standard**
2. **Trade-off balance**:
    - Word-level: Small sequences, huge vocabulary (expensive)
    - Character-level: Tiny vocabulary, huge sequences (expensive)
    - Subword-level: Moderate vocabulary, moderate sequences (optimal)
3. **BPE provides vocabulary size control** through iterative merging
4. **All modern LLMs use BPE or variations** (GPT-2, GPT-3, LLaMA 3, etc.)
5. **Vocabulary sizes in practice**: 50K-200K tokens for modern LLMs
6. **Handles unknown words gracefully**: Breaks into known subword units

### Part 4 
<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Practical Tokenization \& LLM Architecture Summary

## Using Pre-trained Tokenizers (TikToken Library)

### Overview

- **No need to implement tokenization from scratch**
- Pre-trained tokenizers are freely available as open-source libraries
- Pre-trained tokenizers already have vocabularies built from internet data


### TikToken Library (OpenAI)

- **Publisher**: OpenAI
- **Repository**: Available on GitHub with full code
- **Speed**: 3-6x faster than other open-source tokenizers
- **Ease of use**: Very simple Python library


### Installation \& Setup

```python
# Install via pip
pip install tiktoken

# Import library
import tiktoken

# Get tokenizer for specific model
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
```


### Basic Usage

#### Encoding (Text → Numbers)

```python
tokenizer.encode("I love machine learning")
# Output: [1234, 3021, 5678, 9101]
```


#### Decoding (Numbers → Text)

```python
tokenizer.decode([1234, 3021, 5678, 9101])
# Output: "I love machine learning"
```


#### Check Vocabulary Size

```python
tokenizer.vocab_size
# Output: 100,257 (example size)
```


#### Switch Between Models

```python
# GPT-3.5 tokenizer
tokenizer_35 = tiktoken.encoding_for_model("gpt-3.5-turbo")

# GPT-4 tokenizer (newer, likely different vocabulary)
tokenizer_4 = tiktoken.encoding_for_model("gpt-4o")
tokenizer_4.vocab_size  # Example: 200,019 (larger vocabulary)

# Same input may produce different token IDs due to different vocabularies
# tokenizer_35.encode("I love machine learning") ≠ tokenizer_4.encode("I love machine learning")
```


***

## Neural Networks \& LLM Architecture Basics

### What is a Neural Network?

- **Goal**: Learn mapping from input (x) to output (y)
- **Process**: Sequence of learnable transformations
- **Structure**: Input → [Transformations] → Output
- **Training**: Model learns internal parameters (weights) to optimize this mapping


### Three Example Tasks

1. **House price prediction**: Input (location, bedrooms, size) → Output (price)
2. **Email spam detection**: Input (email content) → Output (spam/not spam: 1 or 0)
3. **Tumor detection**: Input (image/pixels) → Output (tumor coordinates: x1, y1, x2, y2)

***

## Linear Layer (Building Block)

### Definition

- Most basic neural network layer
- Performs linear transformation: **y = Wx + b**
    - W = weight matrix (learned parameters)
    - b = bias vector (learned parameters)
    - x = input values


### Simple Example

- **Input**: 4 numbers (0.5, 1, 0.5, 2)
- **Weight matrix**: 1×4 (4 weights for 4 inputs)
- **Bias**: 1 number
- **Output**: 1 number (weighted sum + bias)


### Visual Representation (Neural Network Analogy)

- **Inputs** = "neurons" (circles with values)
- **Connections** = weights between neurons
- **Edges** = represent weight values
- **Outputs** = result neurons


### Generalization

- If inputs have more values → outputs can have more values
- Weight matrix grows proportionally
- Multiple outputs = multiple rows in weight matrix


### Multiple Layers

- **Layer 1**: Transforms input (x) → intermediate representation (x₂)
- **Layer 2**: Transforms x₂ → another representation (x₃)
- **Layer N**: Final transformation → output (y)
- **Overall**: Neural network is a chain of mathematical expressions

***

## Common Neural Network Layers

Beyond linear layers, neural networks use:

- **Convolution layers**: Optimized for spatial data (images)
- **Activation layers**: Add non-linearity to enable learning complex patterns
- **Attention layers**: Enable relationships between different parts of input
- Many others, each with different purposes and strengths


### Key Insight

All layers differ in how they **formulate their transformation**, but they all serve the purpose of transforming input to output in learnable ways.

***

## Transformer Architecture

### Historical Context (2017)

- **Paper**: "Attention Is All You Need" (Google Brain)
- **Innovation**: Novel combination of layers that works exceptionally well for sequence tasks
- **Original task**: Machine translation (text from one language to another)


### Transformer Structure

- **Encoder**: Left side (encodes source language text)
- **Decoder**: Right side (decodes encoded text to target language)
- **Architecture**: Unique stacking of multiple transformer blocks
- **Key components**: Attention layers, linear layers, and other specialized layers


### Why It's Powerful

- Handles sequential data naturally
- Can learn long-range dependencies between tokens
- Highly parallelizable and efficient
- Foundation for all modern LLMs


### Decoder-Only Transformer (For Text Generation)

- **Used for**: Text generation tasks
- **Structure**: Only the decoder part of original transformer
- **Stacked**: Multiple decoder blocks stacked on top of each other
- **Result**: Excellent for next-token prediction


### Modern LLMs Use Decoder-Only Transformers

- All contemporary LLMs are based on decoder-only transformer architecture
- Trained on internet data
- Variations differ only in **hyperparameters** (not architecture)

***

## LLM Architecture (Practical Implementation)

### Input-Output Flow for Text Generation

**Goal**: Model should predict next token

#### Step 1: Tokenization

- Input text: "hi how are"
- Tokenized:  (token IDs)


#### Step 2: Embedding Layer

- **Purpose**: Convert token IDs → vectors
- **Process**: Each ID maps to a learnable embedding vector
- **Why**: Transformer expects sequences of vectors, not numbers
- **Output**: 3 vectors (one per token)


#### Step 3: Transformer Blocks

- **Input**: 3 vectors
- **Process**: Series of transformations
- **Output**: 3 vectors (same length as input)


#### Step 4: Extract Final Vector \& Linear Layer

- **Keep**: Only the last output vector
- **Discard**: First vectors
- **Linear transformation**: Maps vector to vocabulary-size vector
- **Example**: If vocabulary has 50,000 tokens → output vector has 50,000 values


#### Step 5: Softmax \& Probabilities

- **Output interpretation**: Each value = probability of that token being next
- **Vector length**: Equals vocabulary size
- **Sum**: All probabilities sum to 1.0


### Example Probability Distribution

Input: "I hope you"
Output probabilities:

- Token ID 0 (end-of-sentence): 2%
- Token ID 1 (other): 11%
- **Token ID 2 (well)**: **86%** ← Most likely next token
- Others: remaining %

***

## Model Variations (Hyperparameter Differences)

### GPT-2 (OpenAI)

- **Smallest model**: 12 transformer blocks, dimension 768 → ~125M parameters
- **Largest model**: ~1,500,000,000 parameters (1.5 billion)
- **Hyperparameters adjusted**: Number of layers, vector dimensions


### GPT-3 (OpenAI)

- **Same scaling approach** as GPT-2
- **Largest model**: 175,000,000,000 parameters (175 billion)
- **Hyperparameter example**: 96 layers, dimension ~12,288


### LLaMA 3 (Meta)

- **Largest model**: 405,000,000,000 parameters (405 billion)
- **Same architecture**: Decoder-only transformer
- **Hyperparameter example**: Multiple variations with different layer/dimension counts


### General Pattern

- More parameters = greater capacity to learn complex patterns
- More parameters = more expensive to train
- Increase parameters by: Adding more layers, increasing vector dimensions, or both

***

## Model Training Process

### The Problem

- New untrained model has **random parameters**
- Random parameters → random output probabilities
- Cannot rely on random predictions
- **Solution**: Training algorithm to tune parameters


### Training Steps (Repeated Many Times)

#### Step 1: Forward Pass

1. Sample text from internet data
    - Example: "Albert Einstein was a German born"
2. Pass to model
3. Model outputs probabilities for next token
4. Compare with **known correct next token**: "physicist"

#### Step 2: Loss Calculation

- **Loss function**: Cross-entropy loss (standard for classification)
- **Purpose**: Measure how far predictions are from correct answer
- **High loss**: Probabilities not aligned with correct token
- **Low loss**: Probabilities match correct token


#### Step 3: Backpropagation \& Optimization

- **Optimizer algorithm**: Updates all model parameters
- **Goal**: Minimize loss
- **Result**: Next time same input is seen, model predicts more accurately


#### Step 4: Repeat

- Sample different text from training data
- Repeat steps 1-3 many times
- Eventually: Model learns accurate probabilities


### Outcome of Training

- Model learns **token dependencies and statistics** from internet
- Model develops **implicit knowledge of the world**
- Can predict next token accurately based on context
- Example: "I like machine learning because..." → predicts relevant ML-related tokens

***

## Training Challenges at Scale

### Resource Requirements (Example: 405B parameter model)

#### Memory

- Store all model parameters: **1.6 terabytes** (assuming FP32)
- Store intermediate activations during training
- **Not practical on single GPU** (best GPUs ~400GB)
- **Minimum GPUs needed**: 880-1,000 GPUs
- **Realistic**: 2,000+ GPUs required


#### Storage

- **Checkpoint storage**: 2-5 terabytes per checkpoint
- Save checkpoints frequently during training
- Requires massive distributed storage systems


#### Engineering Challenges

- Distributed training across thousands of GPUs
- Network communication overhead
- Synchronization complexity
- Cost: Hundreds of millions of dollars for largest models


### Key Insight

The **algorithm is conceptually simple** (loss → optimization), but **engineering at scale is extremely complex** and expensive.

***

## Summary of LLM Pipeline

1. **Tokenization**: Text → Token IDs
2. **Embedding**: Token IDs → Vectors
3. **Transformer Blocks**: Series of transformations
4. **Final Linear Layer**: Vector → Probability distribution
5. **Training**: Cross-entropy loss + optimization to learn parameters
6. **Inference**: Pass text, get predicted next token probabilities
7. **Generation**: Repeat step 6 to generate multiple tokens

All modern LLMs follow this same architecture—differences are only in hyperparameters (layer count, dimensions, etc.).
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: paste.txt

