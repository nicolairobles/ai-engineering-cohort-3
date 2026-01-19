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

