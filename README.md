# MedAssist AI — Agentic RAG Healthcare Q&A Assistant

A small, self-contained project demonstrating a standard **agentic AI +
LLM** architecture applied to a **healthcare Q&A** use case. It is designed
to be easy to run, easy to explain, and easy to extend — a good fit for a
course project, portfolio piece, or seminar demo.

## What this project demonstrates

- **Retrieval-Augmented Generation (RAG):** user questions are matched
  against a curated medical knowledge base using TF-IDF + cosine similarity
  before an answer is generated, so responses are grounded in known content
  rather than hallucinated.
- **Agentic behaviour (not just a chatbot):** the system does not send
  every message straight to an LLM. A `MedAssistAgent` perceives each
  message and *decides* which tool(s) to invoke:
  1. `red_flag_check` — a safety-first tool that screens for emergency
     symptoms (e.g. chest pain, stroke signs, anaphylaxis, suicidal
     ideation) and immediately escalates with emergency guidance,
     bypassing the LLM entirely.
  2. `needs_clarification` — detects overly vague messages (e.g. "sick")
     and asks a follow-up question instead of guessing.
  3. `retrieve_medical_info` + LLM generation — the standard RAG path for
     well-formed health questions.
  4. `get_disclaimer` — appends a safety disclaimer to every non-emergency
     answer.
- **Pluggable LLM backend:** works with OpenAI, Anthropic, **or** a
  built-in offline `MockBackend` that needs no API key — so the full
  pipeline can be run, tested, and graded without any cost or network
  dependency, and swapped to a real model with one environment variable.
- **Conversation memory:** the agent keeps a running transcript of the
  conversation (`ConversationMemory`), a minimal form of agent state.

## Project structure

```
medassist-ai/
├── app.py                  # CLI entry point (interactive chat or --demo)
├── requirements.txt
├── data/
│   └── medical_kb.json     # curated knowledge base (25 conditions, incl. 6 emergencies)
├── src/
│   ├── retriever.py         # TF-IDF RAG retriever
│   ├── tools.py              # agent tools (red-flag check, clarification, disclaimer)
│   ├── llm_backend.py        # OpenAI / Anthropic / Mock backends
│   └── agent.py               # the agentic orchestrator (perceive-decide-act loop)
└── tests/
    └── test_agent.py        # offline pipeline tests (no API key required)
```

## Setup

```bash
cd medassist-ai
pip install -r requirements.txt
```

To use a real LLM instead of the offline mock backend, set **one** of:

```bash
export OPENAI_API_KEY="sk-..."          # uses gpt-4o-mini
# or
export ANTHROPIC_API_KEY="sk-ant-..."   # uses claude-3-5-haiku
```

and additionally `pip install openai` or `pip install anthropic` as needed.
If neither key is set, the app automatically falls back to the offline
`MockBackend`, so it works out of the box with zero configuration.

## Running it

```bash
# Interactive chat
python app.py

# Scripted demo (6 sample queries covering normal Q&A, vague input,
# and an emergency escalation) — good for a quick walkthrough/report screenshot
python app.py --demo
```

## Running the tests

```bash
python -m pytest tests/ -v
# or, without pytest:
python tests/test_agent.py
```

## Example interaction

```
You: I have sudden chest pain and pain going down my left arm
MedAssist [emergency]: 🚨 EMERGENCY WARNING: These symptoms may indicate a
heart attack, which is a life-threatening emergency. Call emergency medical
services immediately, chew an aspirin if not allergic and advised to do so,
and do not drive yourself to the hospital.
```

## Extending this project

- Swap `TfidfVectorizer` for sentence embeddings (e.g.
  `sentence-transformers`) + a vector store (FAISS/Chroma) for semantic
  retrieval.
- Add more tools: appointment scheduling, medication interaction lookup,
  a symptom-duration follow-up question loop.
- Reimplement `MedAssistAgent`'s router using LangChain/LangGraph or a
  function-calling API for a more standard agent-framework version — the
  perceive → decide → act structure maps directly onto those frameworks'
  agent/tool abstractions.
- Expand `medical_kb.json` with a larger, sourced dataset (e.g. MedlinePlus,
  WHO fact sheets) with proper citation of each source.

## Important disclaimer

This is an **educational/demo project**. It is not a certified medical
device, is not validated for clinical use, and must never be used as a
substitute for professional medical advice, diagnosis, or treatment.
