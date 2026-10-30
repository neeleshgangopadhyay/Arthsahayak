# ArthSahayak — Project Log

Oct 30, 2026

## Overview

ArthSahayak is an Agentic RAG (Retrieval-Augmented Generation) system over Indian Economics documents — RBI reports, Economic Surveys, NCERT textbooks, and budget documents. Unlike a plain RAG pipeline (one retrieval pass, one generation pass), it is being built as an *agentic* system: a planner decides what steps to take, queries are enhanced and reformulated before retrieval, retrieval combines multiple search strategies, and the system reflects on its own answers before returning them — all guarded by input/output safety checks. The project is designed to be resume-worthy: every stage is chosen to demonstrate a specific, named technique (hybrid search, RRF fusion, cross-encoder reranking, query enhancement, LLM gateway design, agentic planning with LangGraph) rather than just "a chatbot over PDFs."

## Current Development Stage

**Steps 1–4 are complete. Step 5 (Guardrails) is next.**

| Step | Name | Status |
| --- | --- | --- |
| 1 | Ingestion (PDFs → chunks → embeddings → Chroma) | Done |
| 2 | Hybrid retrieval (dense + BM25 + RRF fusion + reranking) | Done |
| 3 | LLM Gateway (retry, fallback, rate limiting, logging) | Done |
| 4 | Query enhancement (rewrite + multi-query) wired into retrieval | Done |
| 5 | Guardrails (input + output) | Not started |
| 6 | LangGraph agent (planner + chain-of-thought/self-reflection) | Not started |
| 7 | Grounded answer generation with citations | Not started |
| 8 | Full evaluation suite on the complete agent | Not started |
| 9 | Streamlit frontend | Not started |
| 10 | Polish, README, deployment | Not started |

## Change Log

### Step 1 — Ingestion

- **Switched from `sentence-transformers`/PyTorch to FastEmbed (ONNX-based) for embeddings.** Why: Windows Application Control on this machine blocks the PyTorch DLLs, so any torch-based embedding model failed to load. FastEmbed gives the same embedding quality (`BAAI/bge-small-en-v1.5`) without needing torch at all.
- **Fixed a caching bug where Chroma skipped re-embedding chunks whose metadata changed but whose count stayed the same.** Why: after adding a new `stale_risk` field to chunk metadata, the pipeline compared only chunk *count*, not chunk *content*, so the new field never reached the database for unchanged-count documents. Fixed with a `freshly_processed` flag that forces re-embedding whenever a PDF was freshly reprocessed, not just when its chunk count differs.
- **Added a `stale_risk` metadata flag**, applied to the 2018 Ramesh Singh textbook. Why: that book's data (GDP figures, policy rates, etc.) is outdated compared to current RBI/Economic Survey documents, and the retrieval system needed a way to know that at query time.

### Step 2 — Hybrid Retrieval

- **Built a hybrid search pipeline: dense (vector) search + BM25 (keyword) search, combined with Reciprocal Rank Fusion (RRF), then reranked with a cross-encoder.** Why: dense search alone misses exact numbers/terms (e.g. "7.2% repo rate"); BM25 alone misses semantic meaning (e.g. "food inflation" vs "rising prices of vegetables"). Combining both and reranking the fused list gives the best of both.
- **Applied the `stale_risk` penalty at two separate pipeline stages** — once during RRF fusion, once during cross-encoder reranking. Why: each stage re-scores results independently; a penalty applied only at fusion was undone when the reranker re-promoted the stale document back to the top based on pure text relevance.
- **Built an evaluation harness** (`eval_retrieval.py`) measuring hit-rate@5 across 40 hand-written questions (lookup, exact-number, multi-hop, unanswerable types), with an ablation comparing dense-only vs hybrid vs hybrid+rerank. Why: to objectively prove hybrid search improves retrieval, not just assume it.
- **Result: dense-only 71.8% → hybrid 82.1% → hybrid+rerank 87.2% hit-rate@5**, with improvement in every question category.
- **Fixed an eval ground-truth bug**: one question's "correct answer" was marked as a single page, but the real content was spread across 3 pages of the same document. This made a genuinely correct retrieval look like a miss. Fixed by allowing a list of acceptable pages per question.

### Step 3 — LLM Gateway

- **Built a single `call_llm()` entry point for all LLM calls**, instead of calling the Groq API directly from each script. Why: centralizes retry/backoff, a fallback model, rate limiting, and logging in one place, so every future feature (query enhancement, the agent, guardrails) gets these for free instead of reimplementing them.
- **Added retry-with-backoff, but only for transient errors** (rate limits, connection errors, server errors) — never for client-side errors like a malformed request. Why: retrying a malformed request just repeats the same failure 4 times instead of failing fast.
- **Added a fallback model** (`gpt-oss-20b`) that's used automatically if the main model (`gpt-oss-120b`) fails after all retries. Why: keeps the system usable even if one model has an outage or is rate-limited.
- **Added simple rate limiting** (minimum 1 second between calls) to stay within Groq's free-tier limits.

### Step 4 — Query Enhancement

- **Added query rewriting** (turns a casual question into a clean, keyword-rich search query) and **multi-query expansion** (generates several differently-phrased versions of the same question), both routed through the LLM gateway. Why: users ask casual, ambiguous questions ("what's up with food prices?"); rewriting and diversifying the phrasing before retrieval significantly improves recall.
- **Fixed a bug where query rewriting returned an empty string.** Root cause: `gpt-oss-120b` is a reasoning model — it spends part of its token budget "thinking" before writing the visible answer. With a low `max_tokens` limit, it used the entire budget thinking and had nothing left to write the actual answer, with no error raised. Fixed by passing Groq's `reasoning_effort="low"` parameter (keeps reasoning short for simple tasks) and raising `max_tokens` as a safety margin.
- **Fused query-enhancement results into retrieval** using the same RRF idea already used for dense+BM25 fusion — but this time fusing the hybrid-search results from each query *variant* (original question + rewrite + multi-query versions), so a document confirmed by multiple phrasings ranks above one found by only a single lucky phrasing.

## Planned Next Steps

- **Step 5 — Guardrails (input + output).** Input guardrails: reject or redirect off-topic/unsafe questions before they reach retrieval. Output guardrails: check the generated answer is actually grounded in the retrieved documents (not hallucinated) before returning it to the user.
- **Step 6 — LangGraph agent.** Build the planner node that decides what to do with a question (retrieve, re-plan, ask for clarification), add chain-of-thought/self-reflection (the agent checks its own draft answer and can retry), and design the deferred **live Indian Economics news tool** (RSS feed access) as something the planner can choose to call when a question needs current news rather than static documents.
- **Step 7 — Grounded answer generation with citations.** Generate the final answer from retrieved chunks with inline citations (document + page), so every claim traces back to a source.
- **Step 8 — Full evaluation suite on the complete agent.** Extend the current retrieval-only eval (hit-rate@5) to also evaluate the agent's final answers (faithfulness/groundedness, answer correctness), not just retrieval quality.
- **Step 9 — Streamlit frontend.** A minimal chat UI wired to the agent.
- **Step 10 — Polish, README, deployment.** Write up the project clearly (architecture diagram, eval numbers, setup instructions) and deploy it so it's shareable with a live link, not just runnable locally.

## Resume Bullet Points

- Built an Agentic RAG system over Indian Economics documents (RBI reports, Economic Surveys, NCERT textbooks) using LangChain/LangGraph, combining hybrid retrieval, LLM-driven query enhancement, and a custom LLM gateway with retry/fallback logic.
- Designed a hybrid retrieval pipeline (dense embeddings + BM25 keyword search fused via Reciprocal Rank Fusion, cross-encoder reranking) that improved hit-rate@5 from 71.8% (dense-only) to 87.2%, validated with a 40-question evaluation harness across multiple question types.
- Implemented a recency-aware retrieval penalty to suppress outdated source documents at both the fusion and reranking stages, preventing stale data from outranking current sources.
- Built a production-style LLM gateway abstracting all model calls behind a single interface, with exponential backoff retry (on transient errors only), automatic fallback to a secondary model, rate limiting, and structured logging.
- Implemented query rewriting and multi-query expansion, fused across phrasing variants via Reciprocal Rank Fusion, to improve retrieval robustness to how a question is asked.
