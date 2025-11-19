# NeuraWave — INTERPRETER TECH SPEC (V1)

Purpose
-------
This document defines the exact technical requirements for the `/interpret` microservice.  
The service ingests brand + audience + campaign + signal inputs and returns a deterministic InterpretResponse JSON describing clusters, emotional state, intent, friction, and representative phrases. This is the single most important module for the MVP.

High-level flow
---------------
1. Receive POST /interpret with InterpretRequest JSON.
2. Validate and sanitize inputs (including banned phrases).
3. Extract texts for embedding (brand_DNA, campaign_creatives, signals).
4. Generate embeddings (batch).
5. Run clustering on signal embeddings (HDBSCAN or KMeans fallback).
6. For each cluster compute:
   - emotion_valence
   - intent_score
   - friction_score
   - top_phrases (n-grams)
   - representative_text
7. Compute campaign-level insights (resonance_score, improvement_direction).
8. Return InterpretResponse.

Files & modules
----------------
- `app/main.py` — FastAPI entry; mounts routes and middleware.
- `app/schemas/interpret_request.py` — Pydantic models for request.
- `app/schemas/interpret_response.py` — Pydantic models for response.
- `app/services/embeddings.py` — embedding provider wrapper.
- `app/services/vector_store.py` — optional; functions to save/search vectors.
- `app/services/clustering.py` — cluster algorithm + helper utilities.
- `app/services/scoring.py` — functions for emotion/intent/friction scoring.
- `app/services/text_utils.py` — cleaning, normalization, n-grams extraction.
- `app/services/templates.py` — loads message_templates.json (for later).
- `app/services/inference.py` — high-level orchestration of the interpret flow.
- `app/utils/banned_phrases.json` — banned words list.
- `app/examples/sample_request.json` — example payloads.

API
---
POST /interpret
- Auth: `Authorization: Bearer <TOKEN>`
- Content-Type: `application/json`
- Request Body: InterpretRequest
- Response: InterpretResponse
- Status codes:
  - 200 OK — success
  - 400 Bad Request — schema validation failed
  - 401 Unauthorized — missing/invalid token
  - 429 Rate Limited — too many requests
  - 500 Internal Server Error — unexpected failure

Request validation rules
-------------------------
- Required fields per Pydantic models. Respond 400 if missing.
- Strings trimmed; no binary.
- If any banned phrase appears in `creative_message` or `signals` texts → respond 400 with `error: "banned_content_detected"` and list the matched banned phrase(s).
- Log the sanitized payload (do not store raw PII in logs; mask emails and phone numbers).

Embedding module
----------------
`app/services/embeddings.py` responsibilities:
- Provide `get_embeddings(texts: List[str]) -> List[List[float]]`.
- Use OpenAI embeddings if `OPENAI_API_KEY` present; else, use local sentence-transformers fallback.
- Batch requests for performance; limit batch size to 16–32.
- Normalize vectors (L2 norm).
- Retry strategy: exponential backoff with 3 retries.

Vector store (optional)
-----------------------
- `vector_store.py` should support:
  - `upsert(vectors: List[dict])`
  - `query(vector, top_k=10)`
- Implement Pinecone adapter if keys present; else keep in-memory FAISS for the pilot.

Clustering
----------
`app/services/clustering.py`:
- Primary: HDBSCAN with `min_cluster_size=10` (configurable).
- Fallback: KMeans with `n_clusters=3`.
- Output: for each signal return `cluster_id`.
- Also compute centroid vector and cluster-size.

Text processing & top phrases
-----------------------------
`app/services/text_utils.py`:
- `normalize_text(text)` — lowercase, remove URLs, emojis (optionally keep for sentiment), remove boilerplate.
- `extract_ngrams(texts, n=1..3, top_k=5)` — return top phrases by frequency and TF-IDF weight.
- `sentiment_score(text)` — call a small sentiment model (VADER or simple lexicon) to get -1..+1.

Scoring
-------
`app/services/scoring.py` returns cluster-level metrics:
- `emotion_valence` (-1.0 .. +1.0)
  - computed as weighted combination of sentiment_score and embedding distance against emotion centroids (if present).
- `intent_score` (0.0 .. 1.0)
  - counts of purchase or action keywords (buy, ticket, book, register) normalized by text length and scaled by signal tier weights.
- `friction_score` (0.0 .. 1.0)
  - counts of objection keywords (price, time, busy, can’t, refund) normalized similarly.
- `trust_score` (optional) — presence of UGC, endorsements, verified language.

Signal tiers & weights
----------------------
- HIGH (1.0): search_queries, long-form comments, DMs, repeated page visits, add-to-cart, video-watch > 50%
- MEDIUM (0.6): likes, shares, saves, profile visits
- LOW (0.25): impressions, quick scrolls
Make tier weights configurable via `config.py`.

Campaign insights
-----------------
For each provided campaign input compute:
- `resonance_score` (0.0 .. 1.0) — how aligned the creative_message is to the recommended cluster’s dominant emotion. Compute via cosine similarity between creative_message embedding and cluster centroid + penalty for friction.
- `improvement_direction` — short free-text hint: e.g., "reduce friction by addressing price concerns", "shift messaging to identity/status axis".

Output schema (InterpretResponse)
---------------------------------
Follow `app/schemas/interpret_response.py`. Populate:
- clusters: list of ClusterAnalysis objects with representative_text, top_phrases, emotion_valence, intent_score, friction_score.
- campaign_insights: for each campaign input provide resonance_score and improvement_direction.
- brand_alignment_score: aggregate across campaigns (0..1).
- meta: model_version, run_time_ms, warnings (e.g., banned terms detected).

Error handling & logging
------------------------
- Use structured JSON logs with trace_id for each request.
- Do NOT log raw text of signals in production — mask sensitive content (emails, numbers).
- Return helpful error messages; include `trace_id` for debugging.

Performance & scaling
---------------------
- Aim for single-request latency < 3s for small input sizes (<= 30 signals).
- Use async FastAPI endpoints.
- Batch embeddings and cluster locally.
- Rate-limit LLM/Embedding calls to control cost.

Security & privacy
------------------
- Use token-based Authorization.
- Store only hashed or anonymized inputs if persisting.
- Add environment variables: `OPENAI_API_KEY`, `PINECONE_KEY`, `VECTOR_DB_URL`, `SECRET_TOKEN`.
- Add CORS allowlist if the minimal internal UI is used.

Testing
-------
- Unit tests for: text normalization, embedding wrapper (stubbed), clustering fallback, scoring functions, schema validation.
- Integration test: full interpret run using `app/examples/sample_request.json`. Expect non-empty clusters.
- Add `pytest` and CI (GitHub Actions) for run on PR.

Dev notes & acceptance criteria
-------------------------------
- The interpreter must run && pass the integration test with the sample_request.json.
- The /interpret endpoint must return well-formed JSON matching the InterpretResponse schema.
- The coder must include a README in `app/` with run instructions (how to run with Docker + env variables).

cURL example
------------
