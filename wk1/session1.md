<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# summarize this with hierarchical bullets:

Based on the file content, here is a hierarchical summary of the AI Engineering Course Introduction and Logistics session:

***

## **AI Engineering Course Overview**

### **1. Course Introduction**

- **Instructor:** Ali (ML Engineer \& Tech Lead at Adobe)
    - 7 years at Adobe: 3 years building recommendation systems, recent years on Firefly generative AI team (text-to-video, editing)
    - Previously at Google Maps, worked on ML systems for map experience
    - Co-founded Kung Fu: assistive tech for disabilities using ML
    - Co-authored machine learning system design and generative AI system design interview books
    - Teaches at Stanford Extension Program


### **2. Three Core Principles of Course Design**

- **Learn by Doing**
    - Combines passive learning (videos, readings) with active learning (building projects)
    - Real understanding happens through implementation, debugging, and problem-solving
- **Learn the Right Depth**
    - Avoids extremes: not overly mathematical/academic, not superficial
    - Extracts practical knowledge used in real industry systems
    - Balances learning with hands-on application
- **Learn in a Community**
    - Peer collaboration on capstone projects
    - Constructive feedback from instructors and peers
    - Alumni network for lasting connections beyond the course


### **3. Weekly Course Structure (6-Week Journey)**

**Each week includes four components:**

- Real-time AI project deep dive (live)
- Office hours (live)
- Guided learning release (pre-recorded video, 2-3 hours)
- Project release (template with guided instructions)

**Weekly Breakdown:**

#### **Week 1: LLM Foundations**

- Topics: Chatbot architecture, training, evaluation mechanisms
- Focus: Understand fundamentals (GPT-2 as reference model)
- Project: Build simple LLM from scratch
- Note: More challenging, includes low-level implementation


#### **Week 2: Retrieval-Augmented Generation (RAG)**

- Topics: Retrieval systems, nearest neighbor search, prompt engineering, adaptation techniques
- Project: Customer support chatbot with provided datasets


#### **Week 3: Agents** *(Instructor's favorite)*

- Topics: Tool calling, MCP (Model Context Protocol), reasoning loops, React framework
- Project: Build Perplexity-like search engine (web search + LLM)


#### **Week 4: Reasoning and Thinking LLMs**

- Topics: Inference-time scaling, training-time scaling, deep thinking mechanisms
- Most important for current industry applications
- Project: Deep research capability (multiple agents synthesizing information)


#### **Week 5: Multimodal Generation**

- Topics: Image generation (VAEs, GANs, diffusion models), video generation, efficiency techniques
- Project: Multi-modal agent combining text, image, and video generation


#### **Week 6: Capstone Project**

- Free-form project using course learnings
- Can be individual or team-based
- Optional but highly encouraged
- Feedback provided throughout, demo session at end


### **4. Course Logistics \& Support**

**Content Access:**

- Pre-recorded guided learning videos with learning objectives
- Project templates with setup instructions
- Files/resources provided for each week
- Solutions uploaded after deep dives

**Communication Channels:**

- Weekly Q\&A channels (specific to each week topic)
- General discussion channel (capstone ideas, networking)
- Office hours (live questions)
- Email support (logistical issues)

**Time Commitment:**

- Guided learning: ~2 hours per week
- Live sessions: ~2.5 hours (deep dive + office hour)
- Projects: Variable (optional, 0-2+ hours depending on effort)
- **Total: 4-6 hours average per week**


### **5. Key Course Policies \& Tips**

**Important Notes:**

- Content is sequential; follow in order unless already expert in topic
- Week 1 is more challenging/overwhelming than subsequent weeks
- Don't get discouraged; instructor available to help
- Web searches and library lookups are normal and expected
- Projects assume basic Python knowledge
- Models used are lightweight for accessibility

**Prerequisites:**

- Python programming experience (strongly recommended)
- Access to laptop or Google Colab for projects

**Post-Course Access:**

- Full content access continues after 6 weeks
- GitHub repos removed but solutions remain in portal
- Discord community continues
- Can fork repos before removal


### **6. Capstone Project Details**

- **Timing:** Should start thinking about ideas week 1-2
- **Format:** Individual or team-based, any size
- **Flexibility:** Optional, any AI-related topic
- **Support:** Dedicated channel, office hours feedback, presentation demo session
- **Examples:** Previous cohort projects available for inspiration



<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Ok, but what was it was said about the difference between reasoning models vs. thinking models or something like that.

Based on the transcript, here's what was said about **reasoning models vs. thinking models** (or more accurately, **reasoning LLMs vs. non-reasoning LLMs**):

## **Week 4: Reasoning and Thinking LLMs**

### **Key Distinction Explained:**

**Non-Reasoning LLMs (Weeks 1-3):**

- Standard language models that generate responses directly
- Used in weeks 1, 2, and 3 of the course
- Examples: Standard GPT models without reasoning capabilities

**Reasoning LLMs (Week 4):**

- Models that can "think" before answering
- **Most important week** in terms of what's actually being used in AI applications today
- Can spend extra inference compute to try different approaches and attempts before answering


### **Core Concepts Covered:**

1. **Inference-time scaling** - How to make the model spend more compute at inference to improve answers
2. **Training-time scaling** - Traditional approach to improving models
3. **How to push a non-reasoning LLM to reasoning** - Techniques to make standard models "think"
4. **How to make models think deeply** - Methods to enable models to try multiple approaches before responding

### **Critical Point:**

> "This is very important because **a lot of applications that you would see these days are powered by reasoning algorithms, not non-reasoning algorithms**."

> "In week one, week two, week three, our focus is on **non-reasoning LLMs**. This week, we learn about **reasoning**, and **anything that we've learned from week one to week three can be easily replaced** - the non-reasoning LLM can be easily replaced with a reasoning LLM."

### **Project for Week 4:**

**Deep Research Capability** - similar to ChatGPT's deep research feature, which:

- Uses multiple agents (one lead agent, multiple sub-agents)
- Agents search and collect information
- Lead agent synthesizes everything into a comprehensive document
- Can take 10-30 minutes to complete
- **Requires both agents (Week 3) and reasoning LLMs (Week 4)** to work properly

***

So the distinction isn't really "reasoning vs. thinking" but rather **reasoning LLMs vs. non-reasoning LLMs** - where reasoning models can deliberate and try multiple approaches before responding, rather than generating answers directly.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: paste.txt



***

This course is designed as a comprehensive, practical introduction to building generative AI applications, with emphasis on sequential learning, hands-on implementation, and community support.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: paste.txt
