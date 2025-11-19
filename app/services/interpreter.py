"""
app/services/interpreter.py

A skeleton implementation of the Interpreter orchestration logic.
This file is intentionally self-contained and safe — it uses local placeholders
for embedding/clustering/scoring. The hired dev will replace placeholders
with real integrations (OpenAI embeddings, Pinecone vector DB, HDBSCAN, etc).

This module exposes `run_interpretation(request_dict)` which returns a dict
matching the InterpretResponse schema.
"""

from typing import List, Dict, Any
import time
import re
import json
from pathlib import Path

# Paths (adjust if repo structure differs)
BASE = Path(__file__).resolve().parents[1]
BANNED_PATH = BASE / "utils" / "banned_phrases.json"
TEMPLATES_PATH = BASE / "services" / "message_templates.json"
RULES_PATH = BASE / "services" / "inference_rules.json"

# Load static assets (safe synchronous load for MVP)
def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

BANNED_PHRASES = _load_json(BANNED_PATH) or []
TEMPLATES = _load_json(TEMPLATES_PATH) or {}
INFERENCE_RULES = _load_json(RULES_PATH) or {}

# ---- Utility helpers (simple, replaceable) ----
def _normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"http\S+", "", t)  # remove urls
    t = re.sub(r"\s+", " ", t)
    return t

def _contains_banned(text: str) -> List[str]:
    found = []
    t = _normalize_text(text)
    for term in BANNED_PHRASES:
        if not term:
            continue
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", t):
            found.append(term)
    return found

def _simple_sentiment_score(text: str) -> float:
    # VERY simple placeholder sentiment: +1 if contains positive words, -1 if negative words
    pos = ["good", "great", "love", "excited", "win", "best"]
    neg = ["bad", "hate", "terrible", "expensive", "can't", "cant", "refund"]
    s = 0
    t = _normalize_text(text)
    for p in pos:
        if p in t:
            s += 0.3
    for n in neg:
        if n in t:
            s -= 0.4
    return max(-1.0, min(1.0, s))

def _count_keyword_group(text: str, keywords: List[str]) -> int:
    t = _normalize_text(text)
    count = 0
    for k in keywords:
        if k.lower() in t:
            count += 1
    return count

# ---- Core heuristic implementations (replace with ML later) ----
def infer_emotional_state(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Minimal heuristics to infer a dominant emotion + intensity using rules file.
    """
    # Aggregate tier-based counts and raw texts
    texts = [s.get("text", "") for s in signals if s.get("text")]
    concatenated = " ".join(texts)
    sentiment = _simple_sentiment_score(concatenated)
    # Base metrics
    intent_score = 0.0
    friction_score = 0.0
    # keyword buckets from inference rules
    kw = INFERENCE_RULES.get("keyword_signals", {})
    for group, words in kw.items():
        c = _count_keyword_group(concatenated, words)
        if group == "intent":
            intent_score += c * 0.3
        if group == "friction":
            friction_score += c * 0.35
    # Normalize (simple)
    intent_score = max(0.0, min(1.0, intent_score))
    friction_score = max(0.0, min(1.0, friction_score))
    # Determine dominant emotion heuristically
    if intent_score > 0.5:
        dominant = "hunger"
    elif friction_score > 0.45:
        dominant = "fear"
    elif sentiment > 0.2:
        dominant = "hope"
    elif sentiment < -0.2:
        dominant = "pain"
    else:
        dominant = "hesitation"
    intensity = max(0.0, min(1.0, intent_score * 0.6 + abs(sentiment) * 0.4 + (1 - friction_score) * 0.1))
    return {
        "dominant_emotion": dominant,
        "intensity": round(float(intensity), 3),
        "intent_score": round(float(intent_score), 3),
        "friction_score": round(float(friction_score), 3),
        "example_text": texts[0] if texts else ""
    }

def select_archetype(emotion: str, audience_profile: Dict[str, Any]) -> str:
    """
    Map emotion to archetype; use audience hints for overrides.
    """
    mapping = {
        "hesitation": "oracle",
        "hunger": "warrior",
        "fear": "navigator",
        "pain": "shadow",
        "hope": "oracle",
        "identity": "sovereign",
        "boldness": "warrior",
        "confusion": "architect",
        "overwhelm": "architect",
        "indifference": "shadow"
    }
    archetype = mapping.get(emotion, "oracle")
    # override: if psychographic suggests luxury/status, prefer sovereign
    psy = audience_profile.get("psychographics", [])
    if any("luxury" in p.lower() or "high-net-worth" in p.lower() for p in psy):
        if archetype not in ("sovereign", "architect"):
            archetype = "sovereign"
    return archetype

def pick_template(archetype: str, emotion_intensity: float) -> Dict[str, str]:
    """
    Pick a template for the archetype. Very simple selection: first template, with possible intensity-based variant.
    """
    archs = TEMPLATES.get("archetypes", {}) or {}
    candidate = archs.get(archetype, {})
    templates = candidate.get("templates", [])
    if not templates and "additional" in archs:
        templates = archs["additional"].get("templates", [])
    if not templates:
        return {"hook": "", "short_copy": "", "long_copy": "", "visual_prompt": ""}
    # Choose index by intensity (deterministic)
    idx = int(min(len(templates) - 1, max(0, round(emotion_intensity * (len(templates) - 1)))))
    return templates[idx]

def enforce_brand_dna(message: str, brand_dna_text: str) -> str:
    """
    Simple enforcement: ensure message tone words don't violate brand DNA lines.
    This is a placeholder: real enforcement will be rule-based or LLM-based.
    """
    # Example: if brand_dna contains 'no hype', remove exclamation marks and aggressive words
    t = message
    if "no-hype" in brand_dna_text.lower() or "no hype" in brand_dna_text.lower():
        t = re.sub(r"[!]{1,}", ".", t)
        t = re.sub(r"\b(urgent|now|immediately|act now)\b", "consider", t, flags=re.IGNORECASE)
    return t

# ---- Public orchestration function ----
def run_interpretation(request_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function the FastAPI endpoint will call. Accepts dict shaped like InterpretRequest.
    Returns serializable dict matching InterpretResponse.
    """
    start = time.time()
    # Basic validation (lightweight)
    if not isinstance(request_dict, dict):
        return {"error": "invalid_request_format"}
    brand_dna = request_dict.get("brand_dna", "")
    banned = request_dict.get("banned_phrases", [])
    audience = request_dict.get("audience_profile", {})
    campaign_inputs = request_dict.get("campaign_inputs", [])
    signals = request_dict.get("signals", [])

    # 1) quick banned-phrase check on incoming creatives and signals
    matched = []
    for c in campaign_inputs:
        msg = c.get("creative_message", "")
        matched += _contains_banned(msg)
    for s in signals:
        matched += _contains_banned(s.get("text", ""))

    if matched:
        return {
            "clusters": [],
            "campaign_insights": [],
            "brand_alignment_score": 0.0,
            "meta": {
                "model_version": "v1.0-skeleton",
                "runtime_ms": int((time.time() - start) * 1000),
                "warnings": ["banned_content_detected"],
                "banned_matches": list(set(matched))
            }
        }

    # 2) Infer emotional state from signals (simple heuristic)
    inferred = infer_emotional_state(signals)

    # 3) Choose archetype
    archetype = select_archetype(inferred["dominant_emotion"], audience)

    # 4) Build a representative cluster object (MVP: single cluster)
    cluster_obj = {
        "cluster_id": 1,
        "top_phrases": [],  # coder will implement n-gram extraction
        "emotion_valence": inferred.get("intent_score", 0.0) * 0.7 + inferred.get("friction_score", 0.0) * -0.5,
        "intent_score": inferred.get("intent_score", 0.0),
        "friction_score": inferred.get("friction_score", 0.0),
        "representative_text": inferred.get("example_text", "")
    }

    # 5) Pick a template and craft a candidate message
    template = pick_template(archetype, inferred.get("intensity", 0.5))
    # Merge a simple message: combine hook + short_copy and inject minimal context
    short = template.get("short_copy", "")
    hook = template.get("hook", "")
    # Add a micro-personalization if audience name present
    aud_name = audience.get("name", "")
    if aud_name:
        personalized = f"{hook} {short}"
    else:
        personalized = f"{hook} {short}"
    # enforce brand DNA (placeholder)
    final_message = enforce_brand_dna(personalized, brand_dna)

    # 6) campaign insights (simple resonance by keyword overlap)
    insights = []
    for c in campaign_inputs:
        creative = c.get("creative_message", "")
        # simple resonance: count shared words between creative and final_message
        creative_tokens = set(_normalize_text(creative).split())
        final_tokens = set(_normalize_text(final_message).split())
        overlap = len(creative_tokens.intersection(final_tokens))
        resonance = min(1.0, overlap / max(1, len(final_tokens)))
        improvement_hint = "Consider addressing price concerns" if inferred.get("friction_score", 0) > 0.4 else "Shift to identity-focused angle" if inferred.get("intent_score", 0) > 0.4 else "Clarify next step"
        insights.append({
            "channel": c.get("channel"),
            "creative_message": creative,
            "resonance_score": round(float(resonance), 3),
            "improvement_direction": improvement_hint
        })

    # 7) brand alignment score (placeholder aggregate)
    brand_alignment = max(0.0, min(1.0, 1.0 - inferred.get("friction_score", 0.0)))

    response = {
        "clusters": [cluster_obj],
        "campaign_insights": insights,
        "brand_alignment_score": round(float(brand_alignment), 3),
        "meta": {
            "model_version": "v1.0-skeleton",
            "runtime_ms": int((time.time() - start) * 1000),
            "inferred": inferred,
            "selected_archetype": archetype
        }
    }
    return response
