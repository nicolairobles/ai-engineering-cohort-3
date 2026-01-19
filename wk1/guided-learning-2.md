<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# LLM Model Training, Text Generation \& Post-Training Summary

## LLaMA 3 Training Infrastructure

### Architecture

- **Standard**: Decoder-only transformer architecture (same as all modern LLMs)
- **Parameters**: 405 billion parameters
- **Size constraint**: Cannot fit on single machine (requires distributed training)


### Hardware \& Scaling

- **GPUs used**: 16,000 H100 GPUs (among most powerful available)
- **Infrastructure requirements**:
    - Massive distributed storage for model checkpoints
    - Network communication across thousands of GPUs
    - Complex pipeline parallelism \& distributed training techniques
    - Optimization techniques to fit everything in memory


### Engineering Complexity

- **Code simplicity**: Despite engineering difficulty, actual training code is conceptually simple
    - Calculate loss using cross-entropy loss function
    - Use standard optimizer to update model parameters
    - Repeat on entire internet data
- **Reality**: Extensive optimization tricks \& engineering work required to make training possible at scale


### Outcome

Model becomes excellent at predicting next token based on internet data statistics

***

## Text Generation Process

### Foundation

- **Input**: Trained base model (predicts next token probabilities)
- **Output**: Probability vector where each value = probability for each token in vocabulary


### Why Text Generation is Needed

- **Problem**: Model only outputs probabilities, not actual text
- **Solution**: Iterative process to convert probabilities into meaningful text


### Iterative Generation Loop

**Step-by-step example** (Input: "Albert")

1. **First iteration**:
    - Input: "Albert"
    - Model outputs probabilities for next token
    - Algorithm picks token with highest/best probability: "Einstein"
    - Result: "Albert Einstein"
2. **Second iteration**:
    - Input: "Albert Einstein"
    - Model outputs probabilities for next token
    - Algorithm picks next token: "was"
    - Result: "Albert Einstein was"
3. **Continue** iterating until:
    - Model outputs **end-of-sentence token** (special token indicating sentence completion), OR
    - Reach **maximum desired length** (prevent infinite generation)

### Result

Continuation of initial sentence in meaningful, contextually relevant way

***

## Decoding Algorithms (Token Selection Strategies)

### Overview

- **Purpose**: Specify how to choose token from probability distribution
- **Called**: Decoding algorithms or sampling algorithms
- **Question they answer**: Given probability distribution, which token should we pick?


### Two Categories

#### 1. DETERMINISTIC ALGORITHMS (No Randomness)

Same input + same algorithm = same output every time

##### Greedy Search

- **Algorithm**: Always pick highest probability token
- **Example**:
    - Probabilities: r(37%), s(21%), other(42%)
    - Pick: **r** (highest at 37%)
- **Visualization**: Tree where only highest probability branch taken at each step
- **Advantages**:
    - Very simple
    - Efficient
    - Easy to implement
- **Limitations**:
    - **No look-ahead**: Only considers current highest probability
        - May miss better sequences in other branches
        - Suboptimal global solution
    - **Repetitive outputs**:
        - Common high-probability sequences get repeated over and over
        - Example: "I'm not sure if I'll ever be able to walk with my dog" repeats multiple times
        - Model defaults to statistically common phrases
- **Usage**: Rarely used in practice for LLM text generation


##### Beam Search

- **Algorithm**: Keep track of top-k highest probability paths (not just best one)
- **Example with k=3**:
    - Step 1: Track 3 paths with highest cumulative probability
        - "how come" (24% × 36% = 8.64%)
        - "how are" (31% × 91% = 28.21%)
        - "how do" (26% × 63% = 16.38%)
    - Step 2: For each of 3 paths, generate next token probabilities
        - Process all possibilities, keep top 3 paths again
    - Step 3: Continue until certain iterations reached
    - Final: Pick single highest probability path from k candidates
- **Advantages**:
    - Looks ahead by maintaining multiple paths
    - Often discovers better continuations than greedy search
    - Can sometimes choose alternative branch with better overall probability
- **Limitations**:
    - Still suffers from **repetition issues**
    - More computationally expensive than greedy search
- **Usage**: Less common in modern LLMs (been superseded by sampling methods)

***

#### 2. STOCHASTIC ALGORITHMS (With Randomness)

Same input + same algorithm ≠ same output (adds randomness)

- Can run same generation multiple times, get different continuations
- More creative/diverse outputs


##### Multinomial Sampling

- **Algorithm**: Sample tokens according to their probability distribution
- **Example**:
    - Probabilities: r(37%), s(21%), other(42%)
    - Pick: r with 37% chance, s with 21% chance, other with 42% chance
    - Results in randomness
- **Advantages**:
    - Discovers different continuations (not always picking same token)
    - Randomness enables diversity
- **Limitations**:
    - **Major problem**: Occasionally samples very unlikely tokens
    - If run enough times, may pick grammatically incorrect or semantically wrong tokens
    - No filtering of unlikely words
- **Usage**: Not used in practice (too unpredictable)


##### Top-K Sampling ⭐ (Improvement)

- **Algorithm**:

1. Keep only top-K tokens by probability
2. Sample from these K tokens according to their probabilities
3. Discard all other unlikely tokens
- **Example with K=3**:
    - Original probabilities: r(37%), s(21%), hard(2%), other(40%)
    - Keep: top 3 highest (r, s, other)
    - Discard: hard(2%) and any tokens below top 3
    - Sample: only from these 3 tokens
- **Advantages**:
    - Much better than multinomial (filters unlikely tokens)
    - Only considers likely candidates
- **Limitations**:
    - **K is fixed**: Doesn't adapt to model confidence
    - **Problem scenario 1** (High confidence):
        - Model very confident: "tank" has 89% probability
        - But K=3 still considers top 3 tokens
        - Wastes consideration on unlikely options
    - **Problem scenario 2** (Low confidence):
        - Model uncertain: tokens have similar probabilities
        - More tokens deserve consideration
        - But K=3 limits options
    - **Bottom line**: Fixed K doesn't match varying model confidence


##### Top-P Sampling (Nucleus Sampling) ⭐⭐ STANDARD IN MODERN LLMs

- **Algorithm**:

1. Pick top K tokens where **cumulative probability ≥ P** (hyperparameter)
2. K is **dynamic** (changes based on probability distribution)
3. Sample from this variable-sized set
- **Example with P=0.88 (88%)**:

**Scenario 1** (High confidence):
    - Probabilities: tank(89%), other(11%)
    - Cumulative: tank = 89% (≥ 88%)
    - **Keep**: Only tank
    - Sample: only tank (no uncertainty)

**Scenario 2** (Low confidence):
    - Probabilities: r(31%), s(29%), u(18%), g(10%), other(12%)
    - Cumulative:
        - r = 31% (< 88%)
        - r + s = 60% (< 88%)
        - r + s + u = 78% (< 88%)
        - r + s + u + g = 88% (≥ 88%)
    - **Keep**: r, s, u, g (4 tokens)
    - Sample: from these 4 tokens
- **Key advantage**: K adapts dynamically to model confidence
    - High confidence → keep fewer tokens
    - Low confidence → keep more tokens
- **Usage**: **Standard in modern LLMs** (ChatGPT, Claude, GPT-4, etc.)


### Practical Implementation (Code Example)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Input text
text = "I enjoy walking with my cute dog"
model_inputs = tokenizer(text, return_tensors="pt")

# Top-P sampling
output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,        # Enable sampling
    top_p=0.92,            # Use 92% cumulative probability threshold
    temperature=1.0        # Controls probability distribution shape
)

# Decode output
result = tokenizer.decode(output[^0])
```


### Hyperparameters for Text Generation

#### Temperature

- **Purpose**: Controls smoothness of probability distribution
- **Effect**:
    - Lower temp → sharper/more peaked distribution (more deterministic)
    - Higher temp → smoother/more uniform distribution (more randomness)


#### Top-P Values (Task-Dependent)

| Task | Top-P | Reason |
| :-- | :-- | :-- |
| **Code Generation** | 0.1 | Low diversity needed; focus on syntactically correct code |
| **Factual QA** | 0.3-0.5 | Moderate diversity; need accuracy |
| **Creative Writing** | 0.9 | High diversity; novelty and interesting continuations desired |

**Key insight**: No universally optimal values—must experiment per model and task

***

## Base Models vs. Their Limitations

### What Base Models Do Well

- **Task**: Predict next token (text continuation)
- **Capability**: Has implicit world knowledge from internet data
- **Strength**: Knows relevant terminology for different domains


### Base Model Limitations (Examples)

#### Example 1: Machine Learning Question

**Input**: "I like machine learning because..."

**GPT-2 Base Model**:

- Output: "...it's easy to understand, and you can use it to predict what people are going to say. It's not like you can do that in an elearning system."
- **Problem**: Not answering question, just continuing text with tangential thoughts

**LLaMA 3.1 405B Base Model**:

- Output: "...it's a set of tools that can be applied to a variety of problems. You can use machine learning to predict the price of a house or the probability that someone will click on an ad. You can also use it to identify faces..."
- **Better**: More relevant and coherent continuation with domain knowledge, but still not answering *why* in instructional format


#### Example 2: Direct Question

**Input**: "How is the weather?"

**Base Model Output**: "...weather is one of the most common problems for people who are tired and stressed."

- **Problem**: Not answering the actual question; just continuing as text


### Root Cause

- **Base models are optimized for**: Text continuation based on internet statistics
- **Base models are NOT optimized for**: Following instructions or answering questions
- **Reason**: Pretraining data is raw internet text (not Q\&A format)

***

## Post-Training: Making Models Useful

### Problem

Base model has knowledge but doesn't follow instructions or answer questions → not useful for deployment

### Solution

Post-training has **two stages**:

1. **Supervised Fine-Tuning (SFT)**
2. **Reinforcement Learning (RL)**

***

## Stage 1: Supervised Fine-Tuning (SFT)

### Goal

Adapt base model from **text completion** → **instruction following**

### Process

#### Data Preparation

- **Format**: Question-answer pairs (called "demonstration data")
- **Structure**:

```
INSTRUCTION: "Give three tips for staying healthy"
RESPONSE: "1. Exercise regularly 2. Eat balanced diet 3. Get enough sleep"
```

- **Examples of datasets**:
    - **Alpaca** (Stanford): Open-source instruction tuning dataset
    - **Instruct-GPT** (OpenAI): 14,500 manually curated examples (proprietary)
    - **LIMA** (Meta): High-quality instruction dataset
    - **Math-instruct**: Math-focused demonstrations
    - **No Robots**: General instruction dataset
- **Size**: Tens to hundreds of thousands of examples
    - Much smaller than pretraining data (entire internet)
    - Much higher quality (manually curated by experts)


#### Training Process

- **Algorithm**: Identical to pretraining (no changes)

1. Replace pretraining data with SFT data
2. Calculate cross-entropy loss
3. Use standard optimizer to update parameters
4. Sample from demonstration data
5. Repeat until convergence
- **Example training iteration**:
    - Input: "What is the capital of France"
    - Target: "Paris"
    - Model predicts next tokens
    - Loss calculated, parameters updated


### Result: SFT Model

- **Behavior change**: Now answers questions instead of continuing text
- **Input**: "I want to learn machine learning. What should I do?"
- **SFT Model Output**: "Take Andrew Ng's course on Coursera" (instructional response)
- **vs. Base Model Output**: "...given that it's easy, I'm not so confident as one..." (random continuation)


### Limitation

**SFT Model is NOT ready for deployment** → Need reinforcement learning next

***

## Historical Context

### OpenAI's InstructGPT Paper (March 2022)

- **Paper**: "Training Language Models to Follow Instructions with Human Feedback"
- **Process**:

1. Continued training GPT-3 with SFT
2. Applied reinforcement learning with human feedback
3. Result: InstructGPT (or GPT-3.5)
4. **Chat platform**: ChatGPT launched months later (likely based on refined InstructGPT)


### Why This Matters

- This paper established the standard post-training pipeline used in all modern LLMs
- Same approach used for Claude, Gemini, LLaMA-Instruct, etc.

***

## Key Insights

### Base Model Reality

- **Powerful**: Has world knowledge from billions of internet documents
- **Useless for most tasks**: Only does text continuation
- **Expensive to train**: Months on thousands of GPUs
- **Can't change without full retraining**


### SFT Reality

- **Relatively cheap**: Days of training on hundreds of GPUs
- **Changes behavior fundamentally**: Text continuation → instruction following
- **High-quality data matters**: Expert-curated examples are critical
- **Still incomplete**: Needs RL for safety, alignment, and deployment-readiness

***

## Next Steps

Post-training requires **reinforcement learning** stage to:

- Improve response quality beyond SFT
- Align with human preferences
- Ensure model safety and helpfulness
- Make model production-ready
<span style="display:none">[^1][^2]</span>

<div align="center">⁂</div>

[^1]: paste.txt

[^2]: paste.txt

<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Reinforcement Learning, LLM Evaluation \& System Design Summary

## Reinforcement Learning (RL) Stage Overview

### Problem RL Solves

After SFT, LLM answers questions but with **varying quality levels**

**Example comparison**:

- **Poor response**: "Take Andrew Ng's course on Coursera"
    - Contextually relevant ✓
    - Grammatically correct ✓
    - Answers the question ✓
    - **But**: Too brief, lacks detail, no starting points
- **Excellent response**: "Start with solid foundation in nonlinear algebra, probability, and statistics. Then take a beginner-friendly ML course like Andrew Ng's on Coursera. Practice by building small projects and exploring real datasets."
    - Much more detailed ✓
    - More helpful ✓
    - Provides specific next steps ✓


### RL Goal

Transform SFT model to produce responses that are:

- More detailed
- More accurate
- More correct
- Safer
- More aligned with human preferences

***

## High-Level RL Training Algorithm

### Core Concept: "Practice"

Model learns by practicing (generating multiple responses) and receiving feedback

### Iterative Process

1. **Given a prompt**: "What is 2+2?"
2. **Model generates multiple responses** (e.g., 7 different variations)
3. **Label responses as good/bad**:
    - Green responses = good/preferred
    - Red responses = bad/not preferred
4. **Update model parameters** to reinforce good responses
5. **Repeat**: Model practices again on different prompts

### Result

Model becomes more likely to produce green responses (preferred ones) and less likely to produce red responses (unpreferred ones)

***

## Verifiable vs. Unverifiable Tasks

### Challenge

How do we determine which responses are better than others?

### Two Categories

#### Verifiable Tasks (Objective, Automatic Evaluation)

**Definition**: Can easily verify if response is correct

**Examples**:

- **Math problems**: "What is 2+2?"
    - Correct answer: 4
    - Verification: Automatic (check if final answer = 4)
- **Coding**: "Write a Python function to check if a number is prime"
    - Verification: Run code, check output

**Process**:

1. Generate multiple responses
2. Apply verification logic (Python code/automated checker)
3. Label as correct (green) or incorrect (red)
4. Use PPO algorithm to update model based on labels
5. Model reinforces correct answers

**Result**: Model generates responses with correct final answers

#### Unverifiable Tasks (Subjective, Need Human Input)

**Definition**: Cannot automatically determine correctness (subjective/context-dependent)

**Examples**:

- **Creative writing**: "Write a story about..."
- **Brainstorming**: "Help me choose a good name for my startup"
- **Open-ended advice**: "How should I learn machine learning?"

**Challenge**: No simple way to verify which responses are "correct"

**Solution**: Train a **Reward Model** (separate ML model to score responses)

***

## RLHF: Reinforcement Learning from Human Feedback

### Two-Stage Process for Unverifiable Tasks

#### Stage 1: Training the Reward Model

**Step 1: Data Collection**

- Collect sample prompts (e.g., "What is the capital of France?", "Name a famous physicist")
- Use SFT model to generate multiple responses per prompt
- Example responses for "What is the capital of France?":
    - Response 1: "Paris"
    - Response 2: "It's in Europe"
    - Response 3: "Eiffel Tower"

**Step 2: Human Annotation (Crowdsourcing)**

- Hire human annotators
- Show them all responses for each prompt
- Ask them to **rank responses** (best to worst)
- Example ranking:
    - Best: "Paris" (most accurate)
    - Middle: "It's in Europe" (less accurate)
    - Worst: "Eiffel Tower" (incorrect answer)

**Step 3: Training Data Generation**

- Convert rankings into comparison pairs
- Example training data:

```
Prompt: "What is the capital of France?"
Winning response: "Paris"
Losing response: "It's in Europe"
```

- Create multiple pairs from each ranking

**Step 4: Reward Model Training**

- **Input**: Prompt + Response
- **Output**: Score (numerical rating of response quality)
- **Loss function**: Margin Ranking Loss
    - Maximize difference between winning and losing response scores
    - Increase score of winning responses
    - Decrease score of losing responses

**Result**: Reward model that can **automatically score any response** based on learned human preferences

#### Stage 2: Optimizing the SFT Model with RL

**Process** (nearly identical to verifiable tasks):

1. **Generate responses**: Pass prompt to SFT model → get multiple responses
2. **Score responses**: Reward model scores each response
3. **Apply RL algorithm** (PPO/GRPO):
    - Takes prompt + scores
    - Updates SFT model parameters
    - Reinforces high-scoring responses
4. **Result**: Model generates responses with higher reward scores

### Key Insight

Reward model becomes a **proxy for humans** (instead of annotators ranking each time)

***

## Example: Stress Reduction Question

### Input Prompt

"What are some effective ways to reduce my stress?"

### Possible SFT Responses

1. "Go sky diving for an adrenaline rush" - Not accurate, dangerous
2. **"Exercise regularly and maintain a healthy diet"** - Best answer
3. "Shame on you. Try meditation." - Good advice but impolite/not safe tone
4. "Ignore your problems and hope they go away" - Unhelpful

### After RLHF

Model learns to prefer response \#2 (exercise + diet) over others because:

- Reward model scored it highest (human feedback trained it that way)
- RL training reinforces high-scoring patterns

***

## LLM Training Pipeline Summary

### 4 Stages

| Stage | Task | Data Size | Compute | Duration | Objective |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **Pretraining** | Train on internet | Trillions of tokens | 1,000s of GPUs | Months | Predict next token |
| **SFT** | Instruction following | 10K-100K Q\&A pairs | 100s of GPUs | Days | Predict next token |
| **Reward Modeling** | Learn human preferences | Comparison pairs | 100s of GPUs | Days | Score responses |
| **RL** | Align with humans | 100K-1M prompts | 100s of GPUs | Days | Generate high-reward responses |

### Progression

Base Model → SFT Model → (with Reward Model) → Final Model (deployment-ready)

***

## Practical Comparison: Base vs. Post-Trained Model

### Question

"I want to learn machine learning. What should I do?"

### Base Model (GPT-2) Output

"...where the weather is one of the most common problems for people who are tired and stressed."

- **Problem**: Random continuation, not answering the question


### SFT Model Output

"Take Andrew Ng's course on Coursera."

- **Better**: Answers question
- **Problem**: Still too brief


### Post-Trained Model (LLaMA 3.1 405B) Output

"Learning machine learning is an exciting journey! Here's a step-by-step guide:

- Step 1: Build solid foundation in Python and math programming
- Step 2: [takes care of formatting, bold text, itemized list]
- And so on..."
- **Excellent**: Detailed, helpful, properly formatted, contextually relevant

***

## LLM Evaluation

### Two Types of Evaluation

#### Offline Evaluation (Evaluate on fixed datasets)

**1. Traditional Metrics (Rarely Used)**

- **Metric**: Perplexity
- **What it measures**: How accurately model predicts exact token sequences
- **Example**: Model's probability of generating "how are you doing"
- **Limitation**: Doesn't measure if model actually answers questions correctly
- **Status**: Obsolete for evaluating modern LLMs

**2. Task-Specific Evaluation (Most Common)**

- **Purpose**: Assess performance on real tasks we care about
- **Benchmark types**:
    - **Common Sense Reasoning**: "The trophy doesn't fit in the suitcase because..."
    - **Word Knowledge**: "Who wrote X?" / "Name famous physicist"
    - **Mathematical Reasoning**: "If a train travels 60 mph for 3 hours, how far?"
    - **Code Generation**: "Write a Python function to check if a number is prime"
    - **And many more**: MMLU, Hellaswag, GSM8K, etc.
- **Process**:

1. Run prompt through LLM
2. Compare output to correct answer
3. Score as correct/incorrect
4. Compare across models
- **Advantage**: Directly measures capability on tasks people care about

**3. Human Evaluation**

- **Process**: Ask expert humans to rate LLM responses
- **Advantage**: More nuanced evaluation
- **Disadvantage**:
    - Expensive
    - Can be biased based on evaluator
    - Subjective interpretation
    - Evaluator domain matters


#### Online Evaluation (Real-world deployment)

**1. Human Feedback (In-Production)**

- **Method**: Users give thumbs up/down on responses
- **Example**: ChatGPT's feedback system
- **Data use**:
    - Compare models (which version is better?)
    - Further fine-tuning
    - Reinforcement learning training
- **Advantage**: Real user feedback from actual use cases

**2. Crowdsourcing Platforms**

- **Example**: LMMS (LLM Arena)
    - Open platform developed by graduate students
    - Shows users two anonymous responses from different models
    - Users vote on which is better
    - Aggregates votes to rank models

**Leaderboard Example** (as of writing):

1. Gemini 2.5 Pro (Google) - 1st
2. o3 - 2nd
3. ChatGPT-4o - 3rd
4. GPT-4.5 - 4th
5. Claude+ - 5th
6. DeepSeek (MIT License - open source) - 7th
7. LLaMA 13B - ranked lower (older model)

**Status**: Continuously updated as companies release new models

***

## Complete Chatbot System Architecture

Beyond just the trained LLM, production systems include:

### 1. Input Safety Filter (Guardrails)

- **Purpose**: Ensure input prompt is safe
- **Actions**:
    - Checks for harmful/violent requests
    - If unsafe: generates rejection response ("Sorry, we can't assist with that")
    - If safe: proceeds to next step


### 2. Prompt Enhancer

- **Purpose**: Improve input quality
- **Fixes**:
    - Spelling errors → corrected
    - Grammar issues → corrected
    - Typos → fixed
    - Ambiguities → clarified
- **Methods**: Combination of heuristics + ML models


### 3. Response Generator

- **Purpose**: Generate text response
- **Mechanism**:
    - Takes enhanced prompt
    - Interacts with trained LLM
    - Uses top-P sampling (or other decoding algorithm)
    - Generates tokens one by one
    - Returns response to user


### 4. Session Management

- **Purpose**: Enable multi-turn conversations (follow-ups)
- **How it works**:
    - Maintains chat history of all previous messages
    - **Key insight**: Appends entire conversation history to each new prompt
    - Example flow:

```
User: "Help me name my startup"
LLM sees: [User: "Help me name my startup"]

LLM responds: "How about TechFlow?"

User: "Make it more formal"
LLM sees: [entire history + new message]
→ context enables follow-up understanding
```


### 5. Output Safety Filter (Guardrails)

- **Purpose**: Ensure generated response is safe
- **Actions**:
    - Checks for harmful/biased content
    - If unsafe: uses smaller model to generate rejection
    - If safe: shows response to user


### Workflow

```
User Input 
  ↓
[Input Safety Filter]
  ↓
[Prompt Enhancer]
  ↓
[Response Generator] → LLM interaction
  ↓
[Output Safety Filter]
  ↓
User Sees Response
```

**Note**: Session management integrates throughout—all prior messages sent with each prompt

***

## Week 1 Complete Learning Summary

### Topics Covered

1. **Data Preparation**:
    - Web crawling (Common Crawl)
    - Data cleaning (FineWeb, C4, etc.)
    - Tokenization (BPE algorithm)
2. **Model Architecture**:
    - Neural networks basics
    - Transformers (decoder-only)
    - Next-token prediction
3. **Pretraining**:
    - Train on trillions of tokens
    - Internet data
    - Cross-entropy loss optimization
4. **Text Generation**:
    - Iterative token generation
    - Decoding algorithms (greedy, beam, top-K, top-P)
5. **Post-Training Stage 1 (SFT)**:
    - Supervised fine-tuning on Q\&A pairs
    - Format adaptation: text continuation → question answering
6. **Post-Training Stage 2 (RL)**:
    - Verifiable tasks: automatic scoring
    - Unverifiable tasks: RLHF with reward models
    - PPO algorithm for optimization
7. **Evaluation**:
    - Offline: task-specific benchmarks, human evaluation
    - Online: user feedback, crowdsourcing platforms
8. **System Design**:
    - Safety filters (input/output)
    - Prompt enhancement
    - Session management
    - Production architecture

### Foundation Built

Understanding of complete pipeline from raw internet data → deployment-ready chatbot

### Next Steps

Advanced use cases and extensions (agents, multimodal, etc.)
<span style="display:none">[^1][^2][^3]</span>

<div align="center">⁂</div>

[^1]: paste.txt

[^2]: paste.txt

[^3]: paste.txt

