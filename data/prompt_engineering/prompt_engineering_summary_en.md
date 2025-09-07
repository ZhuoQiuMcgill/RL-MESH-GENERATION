# Prompt Engineering Playbook (Microsoft Learn + OpenAI Help + Anthropic)

Updated: 2025-09-06

This document consolidates prompt engineering best practices from three sources: Microsoft Learn (Azure OpenAI), OpenAI
Help, and Anthropic (Claude 4). It emphasizes clear structure, grounded outputs, and repeatable workflows.

References

- Microsoft Learn: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/prompt-engineering
- OpenAI Help: https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api
- Anthropic (Claude 4): https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices

---

## 1) Cross‑vendor Consensus (What all three agree on)

- Be clear and specific
    - Put the task up front; define goal, audience, scope, constraints, style, tone, language, and length.
    - Prefer “do X” over “don’t do Y”; give alternatives when forbidding behaviors.
- Provide context and grounding
    - Supply domain knowledge, definitions, user profile, date, and authoritative source snippets.
    - The closer the provided material is to the answer form, the fewer errors.
- Use examples (few‑shot)
    - Include 1–3 high‑quality examples covering typical and edge cases; place them consistently.
- Enforce output structure
    - Demand strict JSON/XML/Markdown sections; use delimiters (###, """, XML tags) to separate instruction, context,
      and output.
- Break down complex tasks
    - Stage the work: extract → plan → generate → verify; brief stepwise guidance improves reliability.
- Add stopping conditions
    - Use separators or stop sequences to reduce rambling and off‑task content.
- Tune parameters
    - Lower temperature (≈0–0.3) for factual/extraction; higher (≈0.7–0.9) for creative tasks. Avoid changing
      temperature and top_p simultaneously.
- Evaluate and iterate
    - Define success criteria, curate a test set, and iterate on prompts. Track consistency, factuality, and format
      adherence.
- Reduce hallucinations
    - Require inline citations, restrict to provided sources, and leverage tools/retrieval instead of guessing.

---

## 2) Source‑specific Highlights and Differences

### Microsoft Learn (Azure OpenAI)

- Prompt components: Instructions, Primary Content (target text/data), Examples, Cues (output jumpstarts), Supporting
  Content (metadata/context).
- Ordering and recency: Put the main instruction first; optionally repeat key instructions at the end (recency effect).
- Syntax and delimiters: Use clear section headers and separators (e.g., ---) and uppercase markers for parsing and
  stopping.
- Affordances and chaining:
    - Let the model propose SEARCH(...) or tool calls; then paste results back and re‑ask to verify claims.
    - Encourage chain‑of‑thought style breakdown for reliability and auditability (keep it brief in production).
- API awareness: Distinguish Chat vs Completion styles; provide grounding data; tune temperature and top_p prudently.

### OpenAI Help

- “Rules of thumb” checklist:
    - Use the latest capable model; place instructions first; separate instruction and context with ### or """.
    - Move from zero‑shot → few‑shot → fine‑tune if needed.
    - Demonstrate output format explicitly; reduce fuzzy language (“fairly short”) to concrete constraints (e.g., “3–5
      sentences”).
- Code prompting:
    - Use leading tokens like `import` (Python) or `SELECT` (SQL) to nudge the correct generation pattern.
- Parameters:
    - temperature=0 for deterministic factual tasks; use stop sequences and max token limits to bound outputs.

### Anthropic (Claude 4)

- Explicitness and motivation:
    - Be direct and add “why” (motivation) to improve alignment.
- XML tags and style mirroring:
    - Use XML to strongly steer structure; match prompt style to desired output style to increase steerability.
- Thinking and tools:
    - Support “interleaved thinking” after tool use; prefer parallel tool calls for independent operations.
    - For agentic coding, optionally instruct cleanup of temporary files.
- Frontend/visual generation:
    - Ask explicitly for rich interaction (hover/transition/micro‑interactions) and design principles (
      hierarchy/contrast/balance/movement).
- Avoid “test‑gaming” and hard‑coding:
    - Request general solutions, best practices, and call out infeasible tasks or erroneous tests.
- Migration to Claude 4:
    - Be more explicit about “above‑and‑beyond” expectations and enumerate desired features (e.g., animations,
      interactivity).

---

## 3) Implementation Checklist (Practical steps)

1) Goals and evaluation

- Define success metrics (accuracy, coverage, readability, citation completeness, format validity).
- Prepare a small gold set + ad‑hoc samples for regression.

2) Prompt skeleton (recommended order)

- Role + Task: who the model is and what to do (one sentence).
- Background + Constraints: domain, date, user profile, safety boundaries; if forbidding, specify alternatives.
- Input and Context: wrap source snippets via """/###/XML tags.
- Output Contract: strict format schema, fields, length, language, citation requirements.
- Examples: 1–3 concise examples (typical and edge cases).
- Process Rules: whether to extract→plan→generate or briefly reason first; whether to call tools.

3) Parameters and controls

- temperature: 0–0.2 (factual/extraction); 0.3–0.5 (summaries/rewrites); 0.7–0.9 (creative/brainstorming).
- top_p: typically adjust only one of temperature/top_p.
- stop: specify separators or tokens to end generation.

4) Risks and common pitfalls

- Vague instructions; prohibitions without alternatives; no format contract; missing authoritative context; no inline
  citations.
- Insufficient examples or missing edge cases; no delimiters for long inputs; not leveraging XML tags with Claude.
- “Teaching to the test” and hard‑coding; not asking for quality, generality, and maintainability in code generation.

---

## 4) Ready‑to‑Use Prompt Templates

A) General structured Q&A with JSON output and inline citations

```text path=null start=null
System:
You are a careful, concise assistant.

User:
Task: Summarize the document for a product manager audience in Chinese.
Constraints:
- Length: 5–7 bullet points, each ≤20 words.
- Style: neutral, factual; avoid marketing language.
- Cite inline as [S1], [S2] from provided context only.
Input (triple-quoted):
"""
{{CONTEXT_SNIPPETS}}
"""
Output format (JSON):
{
  "summary": ["...", "...", "..."],
  "open_questions": ["...", "..."]
}
Stop when the JSON is complete.
```

B) Claude/Frontend generation with XML constraints and rich interactions

```text path=null start=null
System:
You are a senior frontend engineer.

User:
Goal: Design a complex interactive dashboard.
<requirements>
- Include hover states, transitions, micro-interactions.
- Apply hierarchy, contrast, balance, movement.
- Go beyond basics; propose 3 polished variants.
</requirements>
<deliverables>
- <wireframes> ...concise description... </wireframes>
- <components> ...list props & states... </components>
- <interactions> ...animation details... </interactions>
- <code> ...React/Tailwind snippets... </code>
</deliverables>
Rules:
- Keep within these tags. Do not use markdown in <code>.
```

C) Information extraction (few‑shot + strict JSON)

```text path=null start=null
System:
Extract structured facts only from the input.

User:
Instruction: Return valid JSON, no extra text.
Schema:
{
  "companies": [string],
  "people": [string],
  "topics": [string],
  "themes": [string]
}
Examples:
Text: "Stripe provides APIs..." -> {"companies":["Stripe"],"people":[],"topics":["APIs","payments"],"themes":["developer tools"]}
Text: "OpenAI has trained..." -> {"companies":["OpenAI"],"people":[],"topics":["language models","text processing"],"themes":["AI platform"]}
Now extract for:
"""
{{TEXT}}
"""
```

D) Code generation (leading token + quality guardrails)

```text path=null start=null
System:
You write high-quality, general-purpose code.

User:
Task: Python CLI that converts miles to km; include tests; no hard-coding.
Quality: readable, robust, maintainable; briefly explain design choices.
Start with:
import
```

E) Grounded answering with inline citations (lower hallucinations)

```text path=null start=null
System:
Answer ONLY from the provided sources, cite inline as [S#].

User:
Question: {{QUESTION}}
Sources:
"""
[S1] ...
[S2] ...
"""
Output:
- 3–5 bullets
- Each claim must include [S#]
If insufficient evidence, respond: "Insufficient support".
```

---

## 5) Parameter Quick Guide

- temperature
    - 0–0.2: extraction, factual Q&A, deterministic formatting
    - 0.3–0.5: summarization, paraphrasing, structured generation
    - 0.7–0.9: creative ideation, marketing copy, brainstorming
- top_p
    - Use as an alternative to temperature; avoid changing both together.
- max tokens (completion)
    - Hard upper bound on generation, not a target length controller.
- stop
    - Stop sequences (e.g., ###, ---) to prevent drifting beyond the contract.

---


## 6) Applications, Importance, and Irreplaceability (evidence‑based)

Applications (real-world industry case studies)
- Financial Services | Morgan Stanley (Wealth Management Internal Reference Assistant)
  - Uses system prompts + templated queries to anchor advisors' questions to internal knowledge bases (investment research, products, and processes), with established "evaluation-iteration (evals)" mechanisms for continuous optimization of prompts and output formats; also launched "AI Debrief" to generate action items and email drafts from meeting notes templates.
- E-commerce/Customer Service | Klarna (AI Customer Service Assistant)
  - Combines role instructions, intent classification, and tool-calling prompts (RAG/function calling) to drive conversations and ticket routing; processed 2.3 million conversations in first month online, covering 2/3 of customer service sessions, equivalent to 700 full-time employees, reducing average handling time from 11 minutes to 2 minutes and repeat inquiries by 25%.
- Education | Khan Academy (Khanmigo Tutoring)
  - Official 7-step prompt engineering methodology (setting teaching role & tone → task breakdown → providing examples → structured output, etc.) drives capabilities like "explain my answer" and "generate lesson plans," representing a classic approach to embedding learning science into system prompts and few-shot examples.
- Healthcare | Clinical "Ambient Scribe"/EHR Drafting
  - Uses department/visit type prompt templates to constrain clinical note structure and terminology (chief complaint-history of present illness-assessment-plan), with subsequent prompt chains to revise tone and coding points; multiple research studies and vendors report significant reduction in documentation time and burnout (Nuance DAX, Epic integration, etc.).
- Public Sector | GOV.UK Chat (UK Government Portal Dialogue Retrieval)
  - Built on RAG foundation, uses government-specific prompt engineering guidelines to standardize security boundaries, refusal strategies, and citation formats; small-scale experiments show most users find it useful with high satisfaction rates.
- Legal | Allen & Overy (A&O Shearman) × Harvey; Thomson Reuters (Westlaw Precision)
  - Law firms and legal research platforms use multi-step agent/tool chain prompts for due diligence, clause comparison, and regulatory Q&A; Westlaw publicly teaches how to write effective legal prompts for verifiable answers with citations.
- Travel/OTA | Expedia (Conversational Trip Planning)
  - Uses intent-slot prompt templates to convert user preferences, budget, duration, etc., into structured conditions for retrieval and comparison, then outputs itineraries and ticket price links with format constraints.
- Retail Platform | Shopify Magic/Sidekick
  - Merchants use editable prompt text in backend to batch-generate product descriptions, theme blocks, and media backgrounds; official documentation explicitly supports "template → adjust prompts" workflow.
- Customer Service Platform | Zendesk (Intelligent Triage & AI Summarization)
  - Leverages prompt templates + intent/sentiment detection for automatic routing, summarization, and macro suggestions, enabling frontline teams to improve first response and resolution rates under established style and format prompts.
- Manufacturing | Siemens Teamcenter + Microsoft Copilot
  - Uses prompt templates in PLM/engineering documents to generate change descriptions, BOM summaries, meeting minutes, and SOP drafts, reducing engineering communication and knowledge lookup costs.
- Gaming | Ubisoft "Ghostwriter" & Roblox Studio Assistant
  - Designers use "intent + tone + scenario" prompt templates to batch-generate NPC "barks" and dialogue drafts; Roblox Assistant/Code Assist uses code context prompts to generate scripts and materials.
- Media/Financial Terminal | Bloomberg (AI News Summary/Document Insights)
  - Uses summary prompt specifications (three key points, terminology preservation, link tracebacks) in terminals to provide key points for long articles and financial reports, improving information consumption efficiency.
- Payments/Developer Ecosystem | Stripe (Support/Risk Control/Documentation)
  - Uses routing and summary prompts to assist customer service, risk assessment, and document retrieval; official case studies and interviews emphasize significantly expanded prompt design space from GPT-3 to GPT-4.

Why prompt engineering is important
- Fast, low‑cost task adaptation: In‑context learning (instructions + few‑shot) achieves competitive performance without dataset collection or fine‑tuning cycles [R1].
- Quality and reliability: Prompt patterns like CoT and self‑consistency materially improve reasoning quality across benchmarks [R2][R3][R4].
- Factuality and traceability: Grounding prompts (RAG + inline citations) reduce hallucinations and provide provenance [R5].
- Safety and compliance: Clear, robust prompts with structure and constraints are required to mitigate prompt injection, over‑reliance, and insecure output handling [R6].

Why prompt engineering is irreplaceable (complements fine‑tuning/tools, not a substitute)
- Instance‑level control: Even with fine‑tuning, each request needs task, audience, style, and safety constraints. Prompts remain the real‑time interface to express user intent per instance [R1].
- Pattern‑dependent capabilities: Many gains (CoT, zero‑shot CoT, self‑consistency) are unlocked by prompt patterns at inference time—not solely by model weights [R2][R3][R4].
- Dynamic knowledge and grounding: For up‑to‑date facts and citations, prompts must orchestrate retrieval and enforce sourcing (RAG), which cannot be baked entirely into parameters [R5].
- Defense‑in‑depth: Securing LLM apps requires prompt‑level guardrails alongside system controls; prompt injection and insecure output handling are inherently prompt‑surface threats [R6].

References
- [R1] Brown et al. (2020). Language Models are Few‑Shot Learners. https://arxiv.org/abs/2005.14165
- [R2] Wei et al. (2022). Chain‑of‑Thought Prompting Elicits Reasoning in Large Language Models. https://arxiv.org/abs/2201.11903
- [R3] Kojima et al. (2022). Large Language Models are Zero‑Shot Reasoners. https://arxiv.org/abs/2205.11916
- [R4] Wang et al. (2023). Self‑Consistency Improves Chain of Thought Reasoning in Language Models. https://arxiv.org/abs/2203.11171
- [R5] Lewis et al. (2020). Retrieval‑Augmented Generation for Knowledge‑Intensive NLP Tasks. https://arxiv.org/abs/2005.11401
- [R6] OWASP (2025). OWASP Top 10 for Large Language Model Applications. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- [R7] Khan Academy Blog (2024/2025). Meet Khan Academy Writing Coach. https://blog.khanacademy.org/meet-khanmigo-writing-coach-helping-learners-become-better-writers/
