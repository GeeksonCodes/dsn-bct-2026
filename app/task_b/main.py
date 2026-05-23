"""
Task B — Recommendation System
FastAPI backend for personalised product recommendations
with Nigerian contextualisation and multilingual support.

Pipeline:
  Cold-start  → elicitation answers → positive-framed query
              → FAISS retrieval → blend scoring (50/50)
              → post-filter → LLM reranker → top 5
  Returning   → user_id → persona extraction → same pipeline
"""

import os
import re
import time
import random
import pickle
import logging
from typing import Optional
from enum import Enum


import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from google import genai  # NOT google.generativeai (deprecated)

# ── LOGGING ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_b")

# ── APP ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Task B — Recommendation System",
    description="Personalised product recommendations with Nigerian contextualisation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════════════════════════

class RecommendRequest(BaseModel):
    user_id: Optional[str] = None
    product_type: Optional[str] = None
    priority: Optional[str] = None
    avoid: Optional[str] = None
    language: str = "english"
    nigerian_mode: bool = True


class RecommendationItem(BaseModel):
    rank: int
    asin: str
    title: str
    avg_rating: float
    review_count: int
    explanation: str = ""


class RecommendResponse(BaseModel):
    reasoning: str
    recommendations: list[RecommendationItem]
    persona_type: str
    language: str


# ══════════════════════════════════════════════════════════════════
#  LANGUAGE PROMPTS  — Nigerian contextualisation
# ══════════════════════════════════════════════════════════════════

LANGUAGE_PROMPTS = {
    "pidgin": {
        "name": "Nigerian Pidgin",
        "greeting": "Wetin you need today?",
        "system": (
            "Respond naturally in Nigerian Pidgin mixed with English.\n"
            "Sound like a trusted friend, not a salesperson.\n"
            'Example: "This one e good o! E no go break your pocket.\n'
            'Many people don use am, dem all happy. I go advise make you try am!"'
        ),
    },
    "yoruba": {
        "name": "Yoruba-English",
        "greeting": "Ẹ káàbọ̀! How can I help you?",
        "system": (
            "Mix Yoruba phrases naturally with English.\n"
            'Example: "Ẹ káàbọ̀! This product o dára gan-an.\n'
            'Ó tọ́ iye rẹ̀ — worth every naira. Ẹ gbìyànjú!"'
        ),
    },
    "hausa": {
        "name": "Hausa-English",
        "greeting": "Sannu! How can I help you?",
        "system": (
            "Mix Hausa phrases naturally with English.\n"
            'Example: "Sannu! Wannan kaya yana da kyau sosai.\n'
            'Mai rahusa ne — affordable and effective. Ina ba da shawara!"'
        ),
    },
    "igbo": {
        "name": "Igbo-English",
        "greeting": "Nnọọ! How can I help you?",
        "system": (
            "Mix Igbo phrases naturally with English.\n"
            'Example: "Nnọọ! Ihe a dị mma nke ọma.\n'
            'O dị ire ire — good value. Nwanne, I ga-amasị ya!"'
        ),
    },
    "english": {
        "name": "Nigerian English",
        "greeting": "Hello! How can I help you today?",
        "system": (
            "Use Nigerian English — practical, value-conscious, communal.\n"
            "Reference everyday Nigerian consumer concerns naturally."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════
#  GLOBALS & MULTI-PROVIDER KEY MANAGER (Section 16)
# ══════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field

class Provider(Enum):
    GEMINI_20_FLASH   = "gemini-2.0-flash"
    GEMINI_15_FLASH   = "gemini-1.5-flash"
    CEREBRAS_DEEPSEEK = "deepseek-r1-distill-llama-70b"
    GROQ_LLAMA        = "groq-llama-3.3-70b"


@dataclass
class KeyStats:
    provider      : Provider
    api_key       : str
    requests_made : int   = 0
    errors_429    : int   = 0
    errors_other  : int   = 0
    total_latency : float = 0.0
    last_used     : float = 0.0
    is_exhausted  : bool  = False

    @property
    def avg_latency(self):
        if self.requests_made == 0:
            return 0.0
        return self.total_latency / self.requests_made

    @property
    def error_rate(self):
        if self.requests_made == 0:
            return 0.0
        return (self.errors_429 + self.errors_other) / self.requests_made

    def to_dict(self):
        return {
            "provider"      : self.provider.value,
            "requests_made" : self.requests_made,
            "errors_429"    : self.errors_429,
            "avg_latency_s" : round(self.avg_latency, 3),
            "error_rate_pct": round(self.error_rate * 100, 1),
            "is_exhausted"  : self.is_exhausted
        }


class MultiProviderKeyManager:
    """
    Manages multiple API keys across multiple providers.
    Automatically rotates on rate limits.
    Tracks performance per key.
    """

    def __init__(self):
        self.keys: list[KeyStats] = []
        self._setup_keys()

    def _setup_keys(self):
        # Gemini primary (with backward-compatible GOOGLE_API_KEY fallback)
        key1 = os.environ.get("GOOGLE_API_KEY_1") or os.environ.get("GOOGLE_API_KEY")
        if key1:
            self.keys.append(KeyStats(Provider.GEMINI_20_FLASH, key1))

        # Gemini secondary
        key2 = os.environ.get("GOOGLE_API_KEY_2")
        if key2:
            self.keys.append(KeyStats(Provider.GEMINI_15_FLASH, key2))

        # Cerebras fallback
        cerebras_key = os.environ.get("CEREBRAS_API_KEY")
        if cerebras_key:
            self.keys.append(KeyStats(Provider.CEREBRAS_DEEPSEEK, cerebras_key))

        # Groq fallback
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            self.keys.append(KeyStats(Provider.GROQ_LLAMA, groq_key))

        if not self.keys:
            # Safe default fallback for development
            logger.warning("No API keys found in GOOGLE_API_KEY_1, GOOGLE_API_KEY, CEREBRAS_API_KEY, or GROQ_API_KEY environment variables.")


    def generate(self, prompt: str) -> Optional[str]:
        """
        Try each provider in order.
        Track latency and errors per key.
        """
        if not self.keys:
            logger.error("No keys available to generate.")
            return None

        for key_stat in self.keys:
            if key_stat.is_exhausted:
                continue

            start = time.time()
            try:
                result = self._call_provider(key_stat, prompt)
                elapsed = time.time() - start

                key_stat.requests_made += 1
                key_stat.total_latency += elapsed
                key_stat.last_used      = time.time()

                return result

            except Exception as e:
                elapsed = time.time() - start
                key_stat.total_latency += elapsed
                key_stat.requests_made += 1

                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                    key_stat.errors_429 += 1
                    logger.warning(f"⚠️  {key_stat.provider.value} rate limited — trying next")
                    if "quota" in err_msg.lower():
                        key_stat.is_exhausted = True
                    else:
                        time.sleep(15)
                else:
                    key_stat.errors_other += 1
                    logger.error(f"❌ {key_stat.provider.value} error: {err_msg[:100]}")

        logger.error("❌ All providers exhausted")
        return None

    def _call_provider(self, key_stat: KeyStats, prompt: str) -> str:
        if key_stat.provider in [
            Provider.GEMINI_20_FLASH,
            Provider.GEMINI_15_FLASH
        ]:
            from google import genai
            client   = genai.Client(api_key=key_stat.api_key)
            response = client.models.generate_content(
                model    = key_stat.provider.value,
                contents = prompt
            )
            return response.text.strip()

        elif key_stat.provider == Provider.CEREBRAS_DEEPSEEK:
            from cerebras.cloud import Cerebras
            client = Cerebras(api_key=key_stat.api_key)
            response = client.chat.completions.create(
                model    = "deepseek-r1-distill-llama-70b",
                messages = [{"role": "user", "content": prompt}],
                max_tokens = 1000,
                temperature = 0.3
            )
            content = response.choices[0].message.content.strip()
            # Clean reasoning <think>...</think> tags if present
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return cleaned

        elif key_stat.provider == Provider.GROQ_LLAMA:
            from groq import Groq
            client = Groq(api_key=key_stat.api_key)
            response = client.chat.completions.create(
                model    = "llama-3.3-70b-versatile",
                messages = [{"role": "user", "content": prompt}],
                max_tokens = 1000
            )
            return response.choices[0].message.content.strip()


    def get_stats(self) -> list[dict]:
        """Returns performance stats for all keys — used by /stats endpoint"""
        return [k.to_dict() for k in self.keys]

    def reset_exhausted(self):
        """Call this daily to reset exhausted keys"""
        for k in self.keys:
            k.is_exhausted = False


# Initialise Globals once
index: faiss.Index = None
item_meta: pd.DataFrame = None
embedder: SentenceTransformer = None
pop_lookup: dict = None
key_manager = MultiProviderKeyManager()


# Use everywhere instead of direct generate() calls
def generate(prompt: str) -> Optional[str]:
    return key_manager.generate(prompt)


# ══════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════

@app.on_event("startup")
def load_artifacts():
    global index, item_meta, embedder, pop_lookup

    # ── FAISS index ──────────────────────────────────────────
    logger.info("Loading FAISS index...")
    index = faiss.read_index("items.index")
    logger.info(f"  Items: {index.ntotal:,}  |  Dim: {index.d}")

    # ── Item metadata ────────────────────────────────────────
    logger.info("Loading item metadata...")
    with open("item_meta.pkl", "rb") as f:
        item_meta = pickle.load(f)
    logger.info(f"  Metadata rows: {len(item_meta):,}")

    # ── Embedder — CPU ONLY, hardcoded ───────────────────────
    logger.info("Loading embedder (all-MiniLM-L6-v2, device=cpu)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # Dimension assertion — index MUST match model (384)
    assert index.d == embedder.get_sentence_embedding_dimension(), (
        f"FATAL: Dimension mismatch — index={index.d}, "
        f"model={embedder.get_sentence_embedding_dimension()}"
    )
    logger.info(f"  Dimension check passed: {index.d}")

    # ── Popularity lookup ────────────────────────────────────
    logger.info("Building popularity lookup...")
    max_reviews = item_meta["review_count"].quantile(0.95)
    item_meta["pop_score"] = (
        item_meta["review_count"].clip(upper=max_reviews) / max_reviews
    ) * item_meta["avg_rating"] / 5.0
    pop_lookup = dict(zip(item_meta["parent_asin"], item_meta["pop_score"]))
    logger.info(f"  Pop lookup: {len(pop_lookup):,} items")

    logger.info(f"  Key Manager Pool initialised with {len(key_manager.keys)} provider(s) ✓")
    logger.info("Task B startup complete ✅")



# ══════════════════════════════════════════════════════════════════
#  RETRIEVAL — FAISS + blended scoring
# ══════════════════════════════════════════════════════════════════

def retrieve_items(
    query_text: str,
    n_results: int = 10,
    min_reviews: int = 3,
) -> list[dict]:
    """
    Embed query → FAISS search → blend 50 % semantic + 50 % popularity.
    Confirmed optimal from ablation: NDCG@10 = 0.4210, Hit@10 = 0.6600.
    """
    query_vec = embedder.encode(
        [query_text], normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_vec, n_results * 3)

    items = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = item_meta.iloc[idx]
        if meta["review_count"] < min_reviews:
            continue

        # Bayesian confidence (dataset mean 3.96)
        prior_rating = 3.96
        prior_count = 10
        conf = (
            (prior_count * prior_rating + meta["review_count"] * meta["avg_rating"])
            / (prior_count + meta["review_count"])
        )

        asin = meta["parent_asin"]
        pop = float(pop_lookup.get(asin, 0.0))

        items.append(
            {
                "asin": asin,
                "title": str(meta["title"]),
                "avg_rating": round(float(meta["avg_rating"]), 2),
                "review_count": int(meta["review_count"]),
                "similarity": float(score),
                "confidence": round(conf, 3),
                "pop_score": pop,
                "final_score": float(score) * 0.5 + pop * 0.5,
            }
        )

    return sorted(items, key=lambda x: x["final_score"], reverse=True)[:n_results]


# ══════════════════════════════════════════════════════════════════
#  QUERY BUILDING — positive framing only
# ══════════════════════════════════════════════════════════════════

def build_retrieval_query(
    product_type: str, priority: str, avoid: str
) -> tuple[str, str]:
    """
    Convert elicitation answers into a retrieval-friendly query.
    POSITIVE FRAMING ONLY — negations confuse embedding search.
    The 'avoid' keyword is returned separately for post-filtering.
    """
    query = (
        f"gentle {product_type} product. "
        f"affordable and good value. "
        f"{priority}. "
        f"natural ingredients. fragrance-free. gentle formula."
    )
    return query, avoid


def post_filter(items: list[dict], avoid_keyword: str) -> list[dict]:
    """Remove items whose title mentions the avoided ingredient."""
    if not avoid_keyword:
        return items
    avoid = avoid_keyword.lower()
    return [item for item in items if avoid not in item["title"].lower()]


# ══════════════════════════════════════════════════════════════════
#  LLM RERANKER — shuffle + retry + hallucination fix + fallback
# ══════════════════════════════════════════════════════════════════

def build_reranker_prompt(
    persona_query: str,
    candidates: list[dict],
    nigerian_mode: bool = True,
    language: str = "english",
) -> tuple[str, list[dict]]:
    """Build reranker prompt.  ALWAYS shuffles candidates first."""
    shuffled = candidates.copy()
    random.shuffle(shuffled)  # ← prevents positional bias

    candidate_list = "\n".join(
        [
            f"{i + 1}. {item['title'][:80]} "
            f"(⭐{item['avg_rating']:.1f}, {item['review_count']} reviews)"
            for i, item in enumerate(shuffled)
        ]
    )

    lang = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["english"])

    nigerian_block = ""
    if nigerian_mode:
        nigerian_block = f"""
{lang['system']}

IMPORTANT: This user is Nigerian. Use Nigerian consumer context —
value consciousness, practical benefits, family/community framing.
Respond in {lang['name']}.
"""

    prompt = f"""
You are a personalised product recommendation agent.

USER PERSONA: {persona_query}
{nigerian_block}
CANDIDATE PRODUCTS:
{candidate_list}

Think step by step about what this user needs, then rerank.

Output EXACTLY in this format — nothing else:

REASONING: [2-3 sentences in {lang['name']}]
RANKING:
1. [product title] | [one sentence why in {lang['name']}]
2. [product title] | [one sentence why in {lang['name']}]
3. [product title] | [one sentence why in {lang['name']}]
4. [product title] | [one sentence why in {lang['name']}]
5. [product title] | [one sentence why in {lang['name']}]
""".strip()

    return prompt, shuffled


def parse_reranker_output(
    raw: str, shuffled_candidates: list[dict]
) -> tuple[str, list[dict]]:
    """
    Parse LLM output with word-overlap hallucination fix.
    Maps hallucinated titles back to real candidates.
    """
    reasoning_match = re.search(
        r"REASONING:\s*(.*?)(?=RANKING:)", raw, re.DOTALL
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    ranking_matches = re.findall(r"\d+\.\s+(.+?)\s*\|\s*(.+)", raw)

    reranked = []
    used_indices = set()

    for llm_title, explanation in ranking_matches:
        best_match = None
        best_score = 0

        for i, candidate in enumerate(shuffled_candidates):
            if i in used_indices:
                continue
            llm_words = set(llm_title.lower().split())
            real_words = set(candidate["title"].lower().split())
            overlap = len(llm_words & real_words)
            if overlap > best_score:
                best_score = overlap
                best_match = (i, candidate)

        if best_match:
            used_indices.add(best_match[0])
            reranked.append(
                {**best_match[1], "explanation": explanation.strip()}
            )

    # If parsing failed completely, fall back to shuffled order
    if len(reranked) == 0:
        reranked = shuffled_candidates[:5]

    return reasoning, reranked


def llm_rerank_safe(
    persona_query: str,
    candidates: list[dict],
    nigerian_mode: bool = True,
    language: str = "english",
    max_retries: int = 3,
) -> dict:
    """
    LLM reranker with:
      ✓ Candidate shuffle (positional bias fix)
      ✓ Retry loop (3 attempts)
      ✓ Word-overlap hallucination fix
      ✓ Rule-based fallback (NEVER crashes)
    """
    prompt, shuffled = build_reranker_prompt(
        persona_query, candidates, nigerian_mode, language
    )

    for attempt in range(max_retries):
        try:
            raw = generate(prompt)
            if raw and "REASONING:" in raw and "RANKING:" in raw:
                reasoning, reranked = parse_reranker_output(raw, shuffled)
                return {"reasoning": reasoning, "reranked": reranked}
            else:
                prompt += "\n\nREMINDER: Must include REASONING: then RANKING:"
        except Exception as e:
            if "429" in str(e):
                time.sleep(15 * (attempt + 1))
            else:
                logger.error(f"Reranker error: {e}")

    # ── RULE-BASED FALLBACK — never crash the endpoint ───────
    reranked = []
    for item in candidates[:5]:
        reranked.append(
            {
                **item,
                "explanation": (
                    f"Strong match — ⭐{item['avg_rating']} "
                    f"from {item['review_count']} buyers."
                ),
            }
        )
    return {
        "reasoning": "Recommended based on relevance and popularity.",
        "reranked": reranked,
    }


# ══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/products")
def get_products(category: str = "all", limit: int = 50):
    """Returns sample products for the review simulator dropdown"""
    sample = item_meta.sample(min(limit, len(item_meta)))
    return {
        "products": [
            {
                "asin" : row["parent_asin"],
                "title": row["title"][:80],
                "rating": round(row["avg_rating"], 1),
                "reviews": int(row["review_count"])
            }
            for _, row in sample.iterrows()
            if len(str(row["title"])) > 10  # filter empty titles
        ]
    }


@app.get("/health")
def health():
    """Health check with evaluation metrics."""
    return {
        "status": "ok",
        "ndcg_at_10": 0.4210,
        "hit_rate_10": 0.6600,
        "items_indexed": index.ntotal if index else 0,
        "dimension": index.d if index else 0,
    }


@app.get("/stats")
def stats():
    """Telemetry stats for the multi-provider key pool."""
    providers_stats = key_manager.get_stats()
    total_reqs = sum(p["requests_made"] for p in providers_stats)
    
    # Active provider is the first one that is configured and not exhausted
    active_p = "none"
    for k in key_manager.keys:
        if not k.is_exhausted:
            active_p = k.provider.value
            break

    return {
        "providers": providers_stats,
        "total_requests": total_reqs,
        "active_provider": active_p
    }



@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Main recommendation endpoint.

    Modes:
      • cold_start     — product_type + priority + avoid provided
      • returning_user — user_id only (falls through to generic persona
                         since this container has no review data)
      • generic        — no inputs → popular recommendations
    """
    language = req.language or "english"
    nigerian_mode = req.nigerian_mode

    # ── DETERMINE MODE ───────────────────────────────────────
    if req.user_id and not req.product_type:
        # Returning user — no review history in this container,
        # so build a generic persona from the user_id.
        persona_query = (
            f"Returning user {req.user_id}. "
            f"Beauty and personal care shopper. "
            f"Values quality and good value for money."
        )
        query = "beauty personal care products good quality effective natural"
        avoid = ""
        persona_type = "returning_user"

    elif req.product_type:
        # Cold-start with elicitation answers
        persona_query = (
            f"User who primarily uses {req.product_type} products. "
            f"Prioritises {req.priority or 'quality'}. "
            f"Avoids {req.avoid or 'nothing specific'}."
        )
        query, avoid = build_retrieval_query(
            req.product_type,
            req.priority or "quality",
            req.avoid or "",
        )
        persona_type = "cold_start"

    else:
        # No user_id and no elicitation — generic recommendations
        persona_query = (
            "Beauty and personal care shopper looking for "
            "popular quality products."
        )
        query = "popular beauty personal care products top rated effective"
        avoid = ""
        persona_type = "generic"

    # ── RETRIEVE ─────────────────────────────────────────────
    candidates = retrieve_items(query, n_results=20)
    filtered = post_filter(candidates, avoid)[:10]
    if not filtered:
        filtered = candidates[:10]

    # ── LLM RERANK ───────────────────────────────────────────
    result = llm_rerank_safe(
        persona_query, filtered, nigerian_mode, language
    )

    # ── FORMAT RESPONSE ──────────────────────────────────────
    recommendations = []
    for i, item in enumerate(result["reranked"][:5], 1):
        recommendations.append(
            RecommendationItem(
                rank=i,
                asin=item["asin"],
                title=item["title"],
                avg_rating=item["avg_rating"],
                review_count=item["review_count"],
                explanation=item.get("explanation", ""),
            )
        )

    return RecommendResponse(
        reasoning=result["reasoning"],
        recommendations=recommendations,
        persona_type=persona_type,
        language=language,
    )
