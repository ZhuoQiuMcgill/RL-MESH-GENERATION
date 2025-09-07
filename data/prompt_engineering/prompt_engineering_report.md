# Prompt Engineering Playbook
> A Comprehensive Guide from Microsoft, OpenAI, and Anthropic

**Last Updated:** 2025-09-06

---

## 📋 Table of Contents
1. [Executive Summary](#executive-summary)
2. [Quick Start Guide](#quick-start-guide)
3. [Core Principles](#core-principles)
4. [Implementation Framework](#implementation-framework)
5. [Parameter Tuning Guide](#parameter-tuning-guide)
6. [Industry Applications](#industry-applications)
7. [Prompt Engineer Requirements](#prompt-engineer-requirements)
8. [Vendor-Specific Guidelines](#vendor-specific-guidelines)
9. [Template Library](#template-library)
10. [References](#references)

---

## Executive Summary

This playbook consolidates prompt engineering best practices from three leading AI providers:
- **Microsoft Learn** (Azure OpenAI)
- **OpenAI** Help Documentation
- **Anthropic** (Claude 4)

### Key Takeaways
- ✅ Start simple: Define role + task with strict output format
- ✅ Use few-shot examples to demonstrate edge cases
- ✅ Stage complex work: Extract → Plan → Generate → Verify
- ✅ Reduce hallucinations with RAG and inline citations
- ✅ Iterate with measurable criteria and regression tests

---

## Quick Start Guide

### 🚀 Three Essential Templates

#### 1. Structured Q&A with Citations
```text
System: You are a careful, concise assistant.

User:
Task: Summarize the document for [AUDIENCE].
Constraints:
- Length: 5-7 bullet points, each ≤20 words
- Style: neutral, factual
- Cite inline as [S1], [S2] from provided context only

Input:
"""
{{CONTEXT}}
"""

Output format (JSON):
{
  "summary": ["...", "..."],
  "open_questions": ["..."]
}
```

#### 2. Information Extraction
```text
System: Extract structured facts only from the input.

User:
Return valid JSON, no extra text.
Schema:
{
  "companies": [string],
  "people": [string],
  "topics": [string]
}

Examples:
Text: "Stripe provides APIs..." 
→ {"companies":["Stripe"],"people":[],"topics":["APIs","payments"]}

Extract from:
"""
{{TEXT}}
"""
```

#### 3. Grounded Answering (Anti-Hallucination)
```text
System: Answer ONLY from provided sources, cite inline as [S#].

User:
Question: {{QUESTION}}
Sources:
"""
[S1] ...
[S2] ...
"""

Output:
- 3-5 bullets
- Each claim must include [S#]
- If insufficient evidence: "Insufficient support"
```

---

## Core Principles

### 🎯 Universal Best Practices (All Vendors Agree)

#### 1. **Clarity & Specificity**
- Put the task up front
- Define: goal, audience, scope, constraints, style, tone, language, length
- Prefer "do X" over "don't do Y"
- Provide alternatives when forbidding behaviors

#### 2. **Context & Grounding**
- Supply domain knowledge, definitions, user profile
- Include authoritative source snippets
- The closer the material to desired format, the fewer errors

#### 3. **Few-Shot Learning**
- Include 1-3 high-quality examples
- Cover both typical and edge cases
- Place examples consistently

#### 4. **Output Structure**
- Demand strict format (JSON/XML/Markdown)
- Use delimiters (###, """, XML tags)
- Separate instruction, context, and output clearly

#### 5. **Task Decomposition**
- Break complex tasks into stages
- Follow: Extract → Plan → Generate → Verify
- Provide stepwise guidance

#### 6. **Control Mechanisms**
- Add stop sequences to prevent rambling
- Set clear boundaries and constraints
- Use separators for parsing

#### 7. **Evaluation & Iteration**
- Define success metrics
- Create test sets with gold standards
- Track: consistency, factuality, format adherence

#### 8. **Hallucination Prevention**
- Require inline citations
- Restrict to provided sources
- Leverage retrieval tools instead of guessing

---

## Implementation Framework

### 📝 Step-by-Step Process

#### Step 1: Define Goals & Metrics
```
✓ Accuracy
✓ Coverage
✓ Readability
✓ Citation completeness
✓ Format validity
```

#### Step 2: Build Prompt Skeleton
```
1. Role + Task (one sentence)
2. Background + Constraints
3. Input/Context (with delimiters)
4. Output Contract (schema, format, length)
5. Examples (1-3 with edge cases)
6. Process Rules (reasoning steps, tool use)
```

#### Step 3: Test & Iterate
```
1. Create gold standard test set
2. Run initial prompts
3. Measure against metrics
4. Identify failure patterns
5. Refine prompts
6. Regression test
```

### ⚠️ Common Pitfalls to Avoid
- ❌ Vague instructions without specifics
- ❌ Prohibitions without alternatives
- ❌ Missing format contracts
- ❌ No authoritative context
- ❌ Insufficient examples
- ❌ No edge case coverage
- ❌ Missing delimiters for long inputs
- ❌ Hard-coding instead of general solutions

---

## Parameter Tuning Guide

### 🎛️ Temperature Settings

| Temperature | Use Case | Examples |
|------------|----------|----------|
| **0.0-0.2** | Factual, Deterministic | • Information extraction<br>• Q&A from documents<br>• Format conversion |
| **0.3-0.5** | Balanced | • Summarization<br>• Paraphrasing<br>• Structured generation |
| **0.7-0.9** | Creative | • Ideation<br>• Marketing copy<br>• Brainstorming |

### 🔧 Other Parameters

| Parameter | Guidelines |
|-----------|------------|
| **top_p** | Use as alternative to temperature, not together |
| **max_tokens** | Hard limit, not target length |
| **stop** | Use sequences like ###, --- to prevent drift |

---

## Industry Applications

### 💼 Real-World Case Studies

#### Financial Services
**Morgan Stanley** - Wealth Management Assistant
- System prompts + templated queries → internal knowledge base
- Evaluation-iteration (evals) mechanism for optimization
- "AI Debrief" generates action items from meeting notes

#### E-commerce
**Klarna** - AI Customer Service
- 2.3M conversations in first month
- Coverage: 2/3 of customer sessions
- Results: 11min → 2min handling time, 25% fewer repeat inquiries

#### Education
**Khan Academy** - Khanmigo Tutoring
- 7-step prompt methodology
- Features: "explain my answer", "generate lesson plans"
- Embeds learning science into system prompts

#### Healthcare
**Clinical Documentation** - Ambient Scribe/EHR
- Department/visit-specific templates
- Structure: Chief complaint → History → Assessment → Plan
- Significant reduction in documentation time and burnout

#### Legal
**Allen & Overy × Harvey** | **Thomson Reuters**
- Multi-step agent/tool chains for due diligence
- Clause comparison and regulatory Q&A
- Verifiable answers with citations

#### Additional Industries
- **Travel (Expedia)**: Intent-slot templates for trip planning
- **Retail (Shopify)**: Batch product description generation
- **Manufacturing (Siemens)**: PLM document generation
- **Gaming (Ubisoft/Roblox)**: NPC dialogue and script generation
- **Media (Bloomberg)**: Financial document summarization
- **Payments (Stripe)**: Support routing and risk assessment

### 📊 Why Prompt Engineering Matters

#### Strategic Value
1. **Fast Adaptation**: In-context learning without fine-tuning [R1]
2. **Quality Improvement**: CoT and self-consistency boost reasoning [R2-R4]
3. **Factuality**: RAG + citations reduce hallucinations [R5]
4. **Safety**: Mitigates prompt injection and insecure outputs [R6]

#### Irreplaceable Role
- **Instance Control**: Real-time intent expression per request
- **Pattern Unlocking**: CoT gains at inference time
- **Dynamic Knowledge**: Up-to-date facts via retrieval
- **Defense-in-Depth**: Prompt-level security guardrails

---

## Prompt Engineer Requirements

### 🎯 Essential Characteristics Breakdown

This checklist clearly separates the "must-have characteristics" of a Prompt Engineer into **Skills** (operational capabilities) and **Knowledge** (required understanding), with "gray areas" and quick assessment suggestions.

### 💪 Skills (What You Can Do)

#### Core Operational Capabilities
1. **Task Decomposition & Instruction Orchestration**
   - Break business objectives into executable subtasks
   - Write step-by-step instructions/checklists

2. **Context Management**
   - Key information placement (front-loading & end reminders)
   - Separator/tag-based organization
   - Long text trimming & key point compression
   - Token budget management

3. **Structured Output Design**
   - Stable JSON/table generation
   - Write JSON Schema/field definitions with validation prompts

4. **Example Engineering (Few-shot)**
   - Select/construct minimal example sets
   - Cover positive cases, negative cases, and edge cases

5. **RAG Prompt Design**
   - Retrieval pre-processing/citation standards
   - "Return NOT_FOUND if not found" fallback strategies

6. **Tool/Function Call Prompting**
   - Tool inventory, trigger conditions, parameter constraints
   - Failure retry and confirmation steps

7. **Security & Protection Prompting**
   - Refusal conditions, sensitive data masking
   - Injection defense (input/output sanitization & minimal permission vocabulary)

8. **Evaluation & Iteration**
   - A/B testing capability
   - Build small test sets with pass rate thresholds
   - Read logs to locate failure cases and refine prompts

9. **Style & Register Control**
   - Role setting
   - Consistent implementation of audience/tone/length/format

10. **Error Analysis & Hallucination Suppression**
    - Identify fabrication sources
    - Add evidence constraints and confidence/uncertainty declarations

11. **Multi-turn/Chain Prompting**
    - Plan-execute-review-reflect pipelines
    - Multi-agent division of labor and interface coordination

12. **Multilingual/Localization Implementation**
    - Bind terminology glossaries/brand style guides and regional specifications in prompts

### 📚 Knowledge (What You Must Understand)

#### Foundational Understanding
1. **LLM Fundamentals & Interfaces**
   - Tokens/context windows
   - Sampling parameters (temperature/top_p)
   - Stop tokens
   - System/user/assistant roles

2. **Model & Product Lineage**
   - Capability boundaries of different models/versions
   - Latency/cost/context length and multimodal features

3. **Retrieval/Embedding Basics**
   - Vector search vs keyword search
   - Recall/relevance
   - Chunking and concatenation strategies

4. **Data & Parsing**
   - JSON/regex/simple syntax trees
   - Common parsing and post-processing patterns

5. **Security & Compliance**
   - OWASP LLM Top-10
   - PII/data minimization
   - Enterprise content and behavior policies

6. **Evaluation Methodology**
   - Human review standards
   - Automated metrics (format pass rate, citation completeness, factual consistency, etc.)

7. **Domain Knowledge**
   - Industry-specific terminology, processes, regulations
   - (Legal/Medical/Financial/Educational/Manufacturing, etc.)

8. **Engineering Integration Basics**
   - API orchestration
   - Timeout/retry/idempotency
   - Observability and log field design

9. **Cost/Performance Trade-offs**
   - Context trimming
   - Batching/concurrency
   - Caching and persistence strategies

### 🔄 Gray Areas (Both Skills & Knowledge)

#### Hybrid Competencies
1. **Prompt Anti-pattern Library (with Countermeasures)**
   - Instruction conflicts
   - Irrelevant detail dilution
   - Over-personification
   - Example answer leakage, etc.

2. **Robustness & Resilience**
   - Adversarial inputs
   - Noise/missing fields
   - Cross-temporal/cross-lingual migration

3. **Human-in-the-Loop (HITL) Processes**
   - When to request human review
   - How to feed human review results back into prompts and test sets

### ✅ Quick Assessment Guide

#### For Skills Assessment:
- **Give a task**: "Convert this business requirement into a 5-step prompt chain"
- **Provide broken output**: "Fix this JSON generation prompt that's producing inconsistent formats"
- **Test security awareness**: "Add injection protection to this customer service prompt"

#### For Knowledge Assessment:
- **Ask about trade-offs**: "When would you use temperature 0 vs 0.7?"
- **Probe model selection**: "Which model for: legal documents vs creative writing?"
- **Check domain understanding**: "What compliance considerations for healthcare prompts?"

#### For Gray Areas:
- **Present edge cases**: "How would you handle multilingual input with missing translations?"
- **Discuss failures**: "This prompt works 90% of the time. How do you diagnose the 10%?"
- **Review processes**: "Design a human review workflow for sensitive content generation"

---

## Vendor-Specific Guidelines

### 🔷 Microsoft Learn (Azure OpenAI)
**Key Concepts:**
- Components: Instructions, Primary Content, Examples, Cues, Supporting Content
- Recency effect: Repeat key instructions at the end
- Use clear delimiters (---, uppercase markers)
- Enable tool calls with SEARCH(...)
- Encourage chain-of-thought reasoning

### 🟢 OpenAI
**Key Concepts:**
- Use latest models first
- Progress: zero-shot → few-shot → fine-tune
- Replace fuzzy language with concrete constraints
- Use leading tokens (`import` for Python, `SELECT` for SQL)
- temperature=0 for deterministic tasks

### 🔵 Anthropic (Claude 4)
**Key Concepts:**
- Be explicit and include "why" (motivation)
- Use XML tags for strong structure steering
- Support parallel tool calls and interleaved thinking
- Request rich interactions for frontend tasks
- Avoid test-gaming and hard-coding
- Enumerate "above-and-beyond" features explicitly

---

## Template Library

### 📚 Advanced Templates

#### Frontend Generation (Claude-Optimized)
```xml
System: You are a senior frontend engineer.

User:
Goal: Design a complex interactive dashboard.
<requirements>
- Include hover states, transitions, micro-interactions
- Apply hierarchy, contrast, balance, movement
- Propose 3 polished variants
</requirements>
<deliverables>
- <wireframes>concise description</wireframes>
- <components>props & states</components>
- <interactions>animation details</interactions>
- <code>React/Tailwind snippets</code>
</deliverables>
```

#### Code Generation with Quality
```text
System: You write high-quality, general-purpose code.

User:
Task: Python CLI that converts miles to km
Requirements:
- Include comprehensive tests
- No hard-coding
- Readable, robust, maintainable
- Explain design choices

Start with:
import
```

#### Multi-Step Reasoning
```text
System: You are a systematic problem solver.

User:
Task: Analyze and solve {{PROBLEM}}

Process:
1. Extract key information
2. Identify constraints
3. Generate solution plan
4. Implement solution
5. Verify correctness

Show your work at each step.
```

---

## References

### 📖 Primary Sources
- [Microsoft Learn - Prompt Engineering](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/prompt-engineering)
- [OpenAI - Best Practices](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)
- [Anthropic - Claude 4 Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)

### 📚 Academic Papers
- [R1] Brown et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [R2] Wei et al. (2022). [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [R3] Kojima et al. (2022). [Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)
- [R4] Wang et al. (2023). [Self-Consistency](https://arxiv.org/abs/2203.11171)
- [R5] Lewis et al. (2020). [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [R6] OWASP (2025). [Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [R7] Khan Academy (2024). [Writing Coach](https://blog.khanacademy.org/meet-khanmigo-writing-coach-helping-learners-become-better-writers/)

---

*End of Document*
