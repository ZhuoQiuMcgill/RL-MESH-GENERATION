# Prompt Engineering: Role, Necessity, Provider Perspectives, Cross-Industry Impact, and the Prompt Engineer Competency Model

> A paper-style synthesis based on guidance from Microsoft, OpenAI, and Anthropic, with documented applications.

**Last Updated:** 2025-09-07

---

## Abstract

Prompt engineering is the practice of expressing intent, context, constraints, and evaluation criteria in
natural-language (and structured) instructions to reliably steer large language models (LLMs) toward desired outcomes.
This paper synthesizes convergent guidance from leading model providers (Microsoft, OpenAI, Anthropic) and consolidates
practical patterns, governance methods, and cross-industry applications. We argue that prompt engineering is necessary
as a first-class product capability because it (i) translates business goals into executable model behaviors without
task-specific training, (ii) improves reasoning and factuality via in-context learning and retrieval, (iii) reduces risk
through explicit constraints and safety guardrails, and (iv) accelerates iteration by enabling measurable evaluation. We
conclude with a competency framework defining the skills and knowledge required of prompt engineers.

Keywords: prompt engineering; in-context learning; chain-of-thought; retrieval-augmented generation; evaluation; safety;
enterprise AI

---

## Table of Contents

1. [Introduction and Strategic Value](#introduction-and-strategic-value)
2. [Conceptual Foundations and Methods](#conceptual-foundations-and-methods)
3. [Perspectives from Major AI Providers](#perspectives-from-major-ai-providers)
4. [Applications Across Industries](#applications-across-industries)
5. [The Prompt Engineer: Skills and Knowledge](#the-prompt-engineer-skills-and-knowledge)
6. [Evaluation, Safety, and Governance](#evaluation-safety-and-governance)
7. [Conclusion](#conclusion)
8. [Appendix A: Prompt Patterns and Templates](#appendix-a-prompt-patterns-and-templates)
9. [Appendix B: Parameterization Cheat Sheet](#appendix-b-parameterization-cheat-sheet)
10. [References](#references)

---

## Introduction and Strategic Value

Prompt engineering operationalizes intent. It is the disciplined process of turning business requirements into precise
instructions that LLMs can execute consistently. Unlike model fine-tuning, prompt engineering works at inference time
and leverages in-context learning [R1], chain-of-thought prompting [R2–R3], and self-consistency [R4] to unlock
reasoning improvements without additional training. When combined with retrieval-augmented generation (RAG) [R5], it
constrains models to authoritative sources, reducing hallucinations and enabling verifiable outputs.

Strategic value at a glance:

- Fast adaptation: In-context learning enables task performance without data labeling or training [R1].
- Quality improvement: Chain-of-thought prompting and self-consistency enhance reasoning [R2–R4].
- Factuality and trust: RAG and inline citations constrain models to authoritative sources and reduce hallucinations [R5].
- Safety and compliance: Explicit refusal and sanitization rules mitigate prompt injection and insecure outputs [R6].
- Dynamic knowledge: Retrieval injects up-to-date facts without retraining.
- Defense-in-depth: Prompt-level guardrails complement system policies and application-layer checks.

---

## Conceptual Foundations and Methods

The providers converge on eight universal principles:

1) Clarity & specificity: Put the task up front; state goal, audience, scope, constraints, tone, and length. Prefer “do
   X” over “don’t do Y”.
2) Context & grounding: Add domain knowledge, definitions, user profile, and source snippets; the closer the context is
   to the desired output, the fewer errors.
3) Few-shot learning: Provide 1–3 high-quality examples covering typical and edge cases.
4) Output structure: Demand strict output formats (JSON/Markdown/XML); use delimiters to isolate instruction, context,
   and outputs.
5) Task decomposition: Stage complex work into Extract → Plan → Generate → Verify.
6) Control mechanisms: Use stop sequences and explicit boundaries to avoid drift.
7) Evaluation & iteration: Define success metrics and test sets; track format adherence, factuality, and consistency.
8) Hallucination prevention: Require inline citations; restrict answers to provided sources; prefer retrieval over
   guessing.

Practical implementation framework:

- Step 1: Define goals & metrics (accuracy, coverage, readability, citation completeness, format validity).
- Step 2: Build a prompt skeleton (role + task; background + constraints; delimited context; output contract; few-shot
  examples; process rules).
- Step 3: Test & iterate (create a gold set; measure; analyze failures; refine; regression-test).

Common pitfalls: vagueness, prohibitions without alternatives, missing output contracts, lack of authoritative context,
insufficient/imbalanced examples, missing delimiters, and hard-coded brittle prompts.

Parameterization and control:

- Temperature: 0.0–0.2 for deterministic extraction and formatting; 0.3–0.5 for balanced summarization/paraphrasing;
  0.7–0.9 for creative ideation.
- Other parameters: top_p as an alternative to temperature (not simultaneously); max_tokens is a hard cap, not a target;
  stop sequences (e.g., ###, ---) to prevent drift.

---

## Perspectives from Major AI Providers

The three providers align on the need for explicit structure, staged reasoning, and measurable evaluation, while
offering distinct emphases:

- Microsoft (Azure OpenAI): Decomposes prompts into components (Instructions, Primary Content, Examples, Cues,
  Supporting Content). Recommends recency effects (repeat key instructions), clear delimiters, tool calls (e.g.,
  SEARCH(...)), and encourages stepwise reasoning.
- OpenAI: Advises using the latest models first; progressing from zero-shot to few-shot to fine-tuning; replacing fuzzy
  language with concrete constraints; leveraging leading tokens (e.g., `import`, `SELECT`); and using temperature=0 for
  deterministic tasks.
- Anthropic (Claude 4): Emphasizes explicitness and motivation (“why”), strong structural tags (e.g., XML), parallel
  tool calls, rich interactions for UI tasks, avoidance of test gaming and brittle hard-coding, and enumerating
  “above-and-beyond” features.

Taken together, major providers converge on a discipline: state intent precisely, structure inputs and outputs, stage
reasoning, integrate tools and retrieval when needed, and measure what matters.

---

## Applications Across Industries

Financial services (Morgan Stanley): Wealth management assistants combining system prompts with templated queries against internal knowledge bases; iterative evaluation (evals) improves robustness; “AI debriefs” convert meeting notes into action items. [S1–S3]

E‑commerce (Klarna): AI customer service at scale (~2.3M conversations in the first month), covering ~2/3 of sessions; handling time reduced (~11 → ~2 minutes) and repeat inquiries decreased (~25%). [S8–S11]

Education (Khan Academy): Khanmigo operationalizes a seven‑step prompting methodology to tutor, explain answers, and generate lesson plans, embedding learning science into system prompts. [S12–S13]

Healthcare (Clinical documentation): Ambient scribing and EHR summarization integrated with Epic; studies report reduced time in notes and after‑hours work and improved same‑day closure. [S33, S34, S36]

Legal (A&O Shearman × Harvey; Thomson Reuters): Multi‑step agent chains for due diligence, clause comparison, and regulatory Q&A; effective legal prompting practices and products emerging. [S18–S20]

Additional industries:
- Travel (Expedia) for conversational trip planning. [S21]
- Retail/Commerce (Shopify) for Magic/Sidekick content creation and admin assistance. [S22–S23]
- Customer service platforms (Zendesk) for intent/language/sentiment triage and routing. [S24–S25]
- Manufacturing/PLM (Siemens Teamcenter × Microsoft) for PLM workflows and copilots. [S26–S28]
- Gaming (Ubisoft Ghostwriter; Roblox Studio Assistant/Code Assist) for NPC dialog and code/content assistance. [S29–S32]
- Media/Finance (Bloomberg) for news/filings summarization and a finance‑tuned LLM (BloombergGPT). [S4–S7]
- Payments (Stripe) for support routing, documentation search, and fraud/risk AI. [S39–S40]

---


## The Prompt Engineer: Skills and Knowledge

Skills (operational capabilities):

1) Task decomposition & instruction orchestration: Break business goals into executable subtasks with stepwise guidance.
2) Context management: Front-load critical details, organize with tags/separators, compress long text, and manage token
   budgets.
3) Structured output design: Produce stable JSON/tables; write schemas and validation prompts.
4) Example engineering: Curate minimal few-shot sets covering positive, negative, and edge cases.
5) RAG prompt design: Define retrieval preprocessing, citation standards, and NOT_FOUND fallbacks.
6) Tool/function calling: Specify tool inventories, triggers, parameters, retries, and confirmations.
7) Security prompting: State refusal conditions, mask sensitive data, and defend against injections via sanitization and
   minimal-permission vocabularies.
8) Evaluation & iteration: A/B tests, small gold sets with pass thresholds, and log-driven failure analysis.
9) Style/register control: Roles, audience, tone, length, and format consistency.
10) Error analysis & hallucination suppression: Identify fabrication sources; require evidence and uncertainty
    statements when needed.
11) Multi-turn chains and multi-agent orchestration: Plan → execute → review → reflect patterns.
12) Multilingual/localization: Bind term glossaries, brand guides, and locale constraints into prompts.

Knowledge (required understanding):

- LLM fundamentals and interfaces: tokens, context windows, sampling, stop tokens, role messages.
- Model and product lineage: capability boundaries, latency/cost trade-offs, context limits, and multimodal features.
- Retrieval and embeddings: vector vs keyword search, recall/relevance, chunking and concatenation.
- Data and parsing: JSON/regex/AST-like patterns; common post-processing.
- Security and compliance: OWASP LLM Top‑10, PII minimization, and enterprise policy alignment [R6].
- Evaluation methodology: human review standards and automated metrics (format pass rate, citation completeness, factual
  consistency).
- Domain expertise: industry terminology, processes, and regulations (e.g.,
  legal/medical/financial/education/manufacturing).
- Engineering integration basics: API orchestration, timeout/retry/idempotency, observability and log design.
- Cost/performance trade-offs: context trimming, batching/concurrency, caching and persistence.

Hybrid (skills + knowledge): anti‑pattern libraries with countermeasures; robustness to adversarial/noisy/missing
inputs; cross-lingual/cross-temporal migration; and human‑in‑the‑loop processes (when to request review; how to recycle
human feedback into prompts and tests).

Assessment suggestions: practical tasks (design a 5‑step prompt chain), repair exercises (stabilize JSON generation),
security drills (add injection defenses), trade‑off questions (temperature 0 vs 0.7), model selection scenarios (legal
vs creative), compliance prompts (healthcare considerations), edge-case handling (multilingual gaps), failure
diagnosis (improving the 10%), and human review workflow design.

---

## Evaluation, Safety, and Governance

Measurable evaluation: Define metrics and test sets early. Track consistency, factuality, format adherence, and citation
completeness. Use regression testing to stabilize behavior across releases.

Security-by-design: Incorporate refusal criteria, sensitive-data masking, input/output sanitization, and
minimal-permission vocabularies. Treat prompt injection as an application‑layer risk and layer defenses accordingly,
aligning with the OWASP Top‑10 for LLM applications [R6].

Operational governance: Log prompts, tool calls, and outcomes with fields enabling reproducibility. Establish change
control for prompts and tests, and integrate reviews (HITL) for sensitive domains.

---

## Conclusion

Prompt engineering is not a temporary workaround; it is a durable interface between human intent and general‑purpose
models. By combining explicit structure, staged reasoning, retrieval grounding, and rigorous evaluation, organizations
can deliver reliable, safe, and cost‑effective AI features without bespoke training. The competency model outlined here
provides a practical baseline for hiring, training, and assessing prompt engineers.

---

## Appendix A: Prompt Patterns and Templates

The following patterns are retained from the original playbook and can be used as building blocks in experiments and
production systems.

1) Structured Q&A with citations

```text
System: You are a careful, concise assistant.

User:
Task: Summarize the document for [AUDIENCE].
Constraints:
- Length: 5–7 bullet points, each ≤20 words
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

2) Information extraction

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

3) Grounded answering (anti‑hallucination)

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
- 3–5 bullets
- Each claim must include [S#]
- If insufficient evidence: "Insufficient support"
```

Additional advanced templates (for generation quality, UI tasks, and multi‑step reasoning):

- Frontend generation (Claude‑optimized)

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
-
<wireframes>concise description</wireframes>
-
<components>props & states</components>
-
<interactions>animation details</interactions>
-
<code>React/Tailwind snippets</code>
</deliverables>
```

- Code generation with quality

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

- Multi‑step reasoning (plan → implement → verify)

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

## Appendix B: Parameterization Cheat Sheet

Temperature settings

| Temperature | Use Case               | Examples                                                      |
|-------------|------------------------|---------------------------------------------------------------|
| 0.0–0.2     | Factual, deterministic | Information extraction; Q&A from documents; format conversion |
| 0.3–0.5     | Balanced               | Summarization; paraphrasing; structured generation            |
| 0.7–0.9     | Creative               | Ideation; marketing copy; brainstorming                       |

Other parameters

| Parameter  | Guidelines                                         |
|------------|----------------------------------------------------|
| top_p      | Use as an alternative to temperature, not together |
| max_tokens | Hard limit, not target length                      |
| stop       | Use sequences like ###, --- to prevent drift       |

---

## References

Primary sources

- Microsoft Learn – Prompt Engineering (Azure
  OpenAI): https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/prompt-engineering
- OpenAI – Best
  Practices: https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api
- Anthropic – Claude 4 Best
  Practices: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices

Academic and technical references

- [R1] Brown et al. (2020). Language Models are Few‑Shot Learners. https://arxiv.org/abs/2005.14165
- [R2] Wei et al. (2022). Chain‑of‑Thought Prompting. https://arxiv.org/abs/2201.11903
- [R3] Kojima et al. (2022). Zero‑Shot Reasoners. https://arxiv.org/abs/2205.11916
- [R4] Wang et al. (2023). Self‑Consistency. https://arxiv.org/abs/2203.11171
- [R5] Lewis et al. (2020). Retrieval‑Augmented Generation. https://arxiv.org/abs/2005.11401
- [R6] OWASP (2025). Top 10 for LLM
  Applications. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- [R7] Khan Academy (2024). Writing
  Coach. https://blog.khanacademy.org/meet-khanmigo-writing-coach-helping-learners-become-better-writers/

Industry case studies

- [S1] OpenAI — Morgan Stanley uses AI evals to shape the future of advisor tools: https://openai.com/index/morgan-stanley/
- [S2] Morgan Stanley — Key milestone in innovation journey with OpenAI: https://www.morganstanley.com/press-releases/key-milestone-in-innovation-journey-with-openai
- [S3] Business Insider — Internal AI solutions to boost employee efficiency (Morgan Stanley Assistant, AI Debrief): https://www.businessinsider.com/internal-artificial-intelligence-solutions-banking-companies-employee-training-2025-5
- [S4] Bloomberg — Launches gen‑AI summarization for news content: https://www.bloomberg.com/company/press/bloomberg-launches-gen-ai-summarization-for-news-content/
- [S5] Institutional Investor — Bloomberg’s first generative AI tool hits the Terminal: https://www.institutionalinvestor.com/article/2cqjgsulkx3md4n3ox2ps/portfolio/bloombergs-first-generative-ai-tool-hits-the-terminal
- [S6] arXiv — BloombergGPT: A Large Language Model for Finance: https://arxiv.org/abs/2303.17564
- [S7] Bloomberg — Introducing BloombergGPT: https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/
- [S8] Klarna — AI assistant handles two‑thirds of chats in first month: https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/
- [S9] PR Newswire — Klarna AI assistant press coverage: https://www.prnewswire.com/news-releases/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month-302072740.html
- [S10] OpenAI — Klarna’s AI assistant case: https://openai.com/index/klarna/
- [S11] Customer Experience Dive — Klarna re‑invests in human talent alongside AI: https://www.customerexperiencedive.com/news/klarna-reinvests-human-talent-customer-service-AI-chatbot/747586/
- [S12] Khan Academy — 7‑step approach to prompt engineering for Khanmigo: https://blog.khanacademy.org/khan-academys-7-step-approach-to-prompt-engineering-for-khanmigo/
- [S13] Khan Academy — Prompt engineering a lesson plan: https://blog.khanacademy.org/prompt-engineering-using-ai-for-effective-lesson-planning/
- [S14] Inside GOV.UK — Findings of first generative AI experiment (GOV.UK Chat): https://insidegovuk.blog.gov.uk/2024/01/18/the-findings-of-our-first-generative-ai-experiment-gov-uk-chat/
- [S15] UK Government — AI Playbook: https://www.gov.uk/government/publications/ai-playbook-for-the-uk-government/artificial-intelligence-playbook-for-the-uk-government-html
- [S16] AI Knowledge Hub — GOV.UK Chat use case: https://ai.gov.uk/knowledge-hub/use-cases/gov-uk-chat/
- [S17] Inside GOV.UK — Private beta of GOV.UK Chat: https://insidegovuk.blog.gov.uk/2024/11/05/were-running-a-private-beta-of-gov-uk-chat/
- [S18] A&O Shearman — Exclusive launch partnership with Harvey: https://www.aoshearman.com/en/news/ao-announces-exclusive-launch-partnership-with-harvey
- [S19] A&O Shearman — Roll out agentic AI agents for complex legal workflows: https://www.aoshearman.com/en/news/ao-shearman-and-harvey-to-roll-out-agentic-ai-agents-targeting-complex-legal-workflows
- [S20] Thomson Reuters — Introduction to writing effective AI legal prompts: https://legal.thomsonreuters.com/blog/writing-effective-legal-ai-prompts/
- [S21] Expedia Group — In‑app conversational trip planning powered by ChatGPT: https://www.expediagroup.com/investors/news-and-events/financial-releases/news/news-details/2023/Chatgpt-Wrote-This-Press-Release--No-It-Didnt-But-It-Can-Now-Assist-With-Travel-Planning-In-The-Expedia-App/default.aspx
- [S22] Shopify — Magic and Sidekick product page: https://www.shopify.com/magic
- [S23] Shopify Help Center — Shopify Magic overview: https://help.shopify.com/en/manual/shopify-admin/productivity-tools/shopify-magic
- [S24] Zendesk — About intelligent triage: https://support.zendesk.com/hc/en-us/articles/4964463770650-About-intelligent-triage
- [S25] Zendesk — Viewing intelligent triage predictions: https://support.zendesk.com/hc/en-us/articles/4685355428250-Viewing-intelligent-triage-predictions
- [S26] Siemens — Partners with Microsoft to deliver AI‑enhanced PLM solutions: https://news.siemens.com/en-us/siemens-xcelerator-microsoft-azure/
- [S27] Microsoft — Siemens and Microsoft partner to drive cross‑industry AI adoption: https://news.microsoft.com/source/2023/10/31/siemens-and-microsoft-partner-to-drive-cross-industry-ai-adoption/
- [S28] Siemens Blog — New Teamcenter copilot capabilities: https://blogs.sw.siemens.com/teamcenter/teamcenter-plm-ai-copilot/
- [S29] Ubisoft — Introducing Ghostwriter: https://news.ubisoft.com/en-us/article/7Cm07zbBGy4Xml6WgYi25d/the-convergence-of-ai-and-creativity-introducing-ghostwriter
- [S30] Game Developer — GDC details on Ubisoft’s narrative AI tools: https://www.gamedeveloper.com/marketing/here-are-more-details-on-ubisoft-s-narrative-ai-tools-from-gdc-2023
- [S31] Roblox Developer Forum — Code Assist full release: https://devforum.roblox.com/t/code-assist-full-release-ai-powered-code-completion/2848978
- [S32] Roblox Creator Hub — Assistant for Studio documentation: https://create.roblox.com/docs/assistant/guide
- [S33] Healthcare Dive — Nuance’s AI clinical scribe integrated with Epic (general availability): https://www.healthcaredive.com/news/nuance-dax-copilot-epic-available-artificial-intelligence-clinical-documentation/705026/
- [S34] JAMA Network Open — Clinician experiences with ambient scribe technology: https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2830383
- [S35] PubMed Central — Impact of Nuance DAX ambient listening AI (cohort study): https://pmc.ncbi.nlm.nih.gov/articles/PMC10990544/
- [S36] Epic — Nuance and Epic expand ambient documentation integration: https://www.epic.com/epic/post/nuance-and-epic-expand-ambient-documentation-integration-across-the-clinical-experience-with-dax-express-for-epic/
- [S37] Microsoft — Microsoft and Epic expand AI collaboration in healthcare: https://blogs.microsoft.com/blog/2023/08/22/microsoft-and-epic-expand-ai-collaboration-to-accelerate-generative-ais-impact-in-healthcare-addressing-the-industrys-most-pressing-needs/
- [S38] STAT — Microsoft to embed AI clinical documentation tool in Epic: https://www.statnews.com/2024/01/18/microsoft-dax-copilot-clinical-notes-epic-health-records/
- [S39] OpenAI — Stripe case study: https://openai.com/index/stripe/
- [S40] Stripe — AI features for payments and fraud: https://stripe.com/payments/ai
