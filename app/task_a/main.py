"""
Task A — User review simulation system
FastAPI backend for simulating reviews conditioned on a user profile (cached or built with Agent 1)
and RAG examples (retrieved via FAISS and filtered to the specific user's reviews).
"""

import os
import re
import time
import json
import random
import pickle
import hashlib
import logging
from typing import Optional, List
from enum import Enum

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from sentence_transformers import SentenceTransformer
from google import genai  # NOT google.generativeai (deprecated)
from google.genai import types

# ── LOGGING ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_a")

# ── APP ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Task A — User Modeling Review Simulator",
    description="Simulate what a user would write as a review for an unseen product",
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
#  SCHEMAS & ENUMS
# ══════════════════════════════════════════════════════════════════

def truncate(value: str, limit: int) -> str:
    """Truncate string to limit, appending ellipsis if cut."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit - 3] + "..."
    return value


class RatingTendency(str, Enum):
    generous = "generous"
    harsh = "harsh"
    balanced = "balanced"


class Consistency(str, Enum):
    consistent = "consistent"
    variable = "variable"
    polarised = "polarised"


class WritingStyle(str, Enum):
    verbose = "verbose"
    moderate = "moderate"
    terse = "terse"


class PriceSensitivity(str, Enum):
    very_high = "very high"
    high = "high"
    moderate = "moderate"
    low = "low"


class SkepticismLevel(str, Enum):
    very_high = "very high"
    high = "high"
    moderate = "moderate"
    low = "low"


class RatingBehaviour(BaseModel):
    tendency: RatingTendency
    consistency: Consistency
    never_gives: List[int] = Field(
        default_factory=list,
        description="STRICT: List only integers 1-5. e.g. [5] or []"
    )
    pattern: str = Field(
        max_length=200,
        description="STRICT CONSTRAINT: Under 200 characters. Use fragments if needed."
    )

    @field_validator('pattern', mode='before')
    @classmethod
    def truncate_pattern(cls, v):
        return truncate(str(v), 200) if v else v

    @field_validator('never_gives', mode='before')
    @classmethod
    def validate_never_gives(cls, v):
        if not isinstance(v, list):
            return []
        return [int(x) for x in v if str(x).strip().isdigit() and 1 <= int(x) <= 5]


class WritingVoice(BaseModel):
    style: WritingStyle
    tone: str = Field(
        max_length=150,
        description="STRICT CONSTRAINT: Under 150 characters. Single descriptive phrase."
    )
    structure: str = Field(
        max_length=250,
        description="STRICT CONSTRAINT: Under 250 characters. How they organise a review."
    )
    signature_phrases: List[str] = Field(
        default_factory=list,
        description="Actual phrases from their reviews. No invented phrases."
    )

    @field_validator('tone', mode='before')
    @classmethod
    def truncate_tone(cls, v):
        return truncate(str(v), 150) if v else v

    @field_validator('structure', mode='before')
    @classmethod
    def truncate_structure(cls, v):
        return truncate(str(v), 250) if v else v

    @field_validator('signature_phrases', mode='before')
    @classmethod
    def clean_phrases(cls, v):
        if not isinstance(v, list):
            return []
        return [str(p)[:100] for p in v if p]


class NigerianSignals(BaseModel):
    price_sensitivity: PriceSensitivity
    scepticism: SkepticismLevel
    community_oriented: bool
    cultural_notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="STRICT CONSTRAINT: Under 1000 characters."
    )

    @field_validator('cultural_notes', mode='before')
    @classmethod
    def truncate_cultural_notes(cls, v):
        if v is None:
            return v
        return truncate(str(v), 1000)

    @field_validator('community_oriented', mode='before')
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ('true', 'yes', '1')
        return bool(v)


class PersonaDossier(BaseModel):
    user_id: str
    core_identity: str = Field(
        max_length=600,
        description="STRICT CONSTRAINT: Under 600 characters."
    )
    rating_behaviour: RatingBehaviour
    writing_voice: WritingVoice
    deep_traits: List[str] = Field(
        min_length=3,
        description="Specific discovered traits."
    )
    what_they_care_about: List[str] = Field(
        min_length=1,
        description="Ranked by importance — most dominant first."
    )
    what_they_ignore: List[str] = Field(default_factory=list)
    context_clues: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="STRICT CONSTRAINT: Under 1000 characters."
    )
    nigerian_signals: NigerianSignals
    simulation_brief: str = Field(
        max_length=1000,
        description="STRICT CONSTRAINT: Under 1000 characters."
    )
    avg_rating: float = 3.0
    rating_std: float = 0.0
    review_count: int = 0

    @field_validator('core_identity', mode='before')
    @classmethod
    def truncate_core_identity(cls, v):
        return truncate(str(v), 600) if v else v

    @field_validator('context_clues', mode='before')
    @classmethod
    def truncate_context_clues(cls, v):
        if v is None:
            return v
        return truncate(str(v), 1000)

    @field_validator('simulation_brief', mode='before')
    @classmethod
    def truncate_simulation_brief(cls, v):
        return truncate(str(v), 1000) if v else v

    @field_validator('deep_traits', mode='before')
    @classmethod
    def clean_deep_traits(cls, v):
        if not isinstance(v, list):
            return []
        return [str(t)[:300] for t in v if t and len(str(t).strip()) > 3]

    @field_validator('what_they_care_about', mode='before')
    @classmethod
    def clean_cares_about(cls, v):
        if not isinstance(v, list):
            return ['product quality']
        return [str(t)[:200] for t in v if t]

    @field_validator('what_they_ignore', mode='before')
    @classmethod
    def clean_ignores(cls, v):
        if not isinstance(v, list):
            return []
        return [str(t)[:200] for t in v if t]

    @field_validator('avg_rating')
    @classmethod
    def valid_rating_range(cls, v):
        if not 1.0 <= v <= 5.0:
            raise ValueError(f"avg_rating {v} out of range 1–5")
        return round(v, 2)

    @model_validator(mode='after')
    def brief_references_identity(self):
        if not self.simulation_brief or len(self.simulation_brief) < 20:
            self.simulation_brief = (
                f"Write as a {self.rating_behaviour.tendency} rater "
                f"with a {self.writing_voice.tone} tone. "
                f"They care most about: {', '.join(self.what_they_care_about[:2])}."
            )
        return self


class ItemCard(BaseModel):
    asin: str = ""
    title: str = ""
    product_summary: str = Field(
        max_length=400,
        description="2-3 sentence plain English description of what this product is"
    )
    key_features: List[str] = Field(
        default_factory=list,
        description="Up to 5 specific product features"
    )
    typical_use_case: str = Field(
        max_length=200,
        description="Who uses this and why"
    )
    price_tier: str = Field(
        default="mid-range",
        description="budget | mid-range | premium"
    )
    brand_notes: str = Field(
        default="",
        max_length=200,
        description="Anything notable about the brand, empty string if unknown"
    )
    
    @field_validator('product_summary', mode='before')
    @classmethod
    def trim_summary(cls, v):
        return str(v)[:400] if v else ""
 
    @field_validator('typical_use_case', mode='before')
    @classmethod
    def trim_use_case(cls, v):
        return str(v)[:200] if v else ""
 
    @field_validator('brand_notes', mode='before')
    @classmethod
    def trim_brand_notes(cls, v):
        return str(v)[:200] if v else ""
 
    @field_validator('price_tier', mode='before')
    @classmethod
    def normalise_price_tier(cls, v):
        v = str(v).lower().strip()
        if 'budget' in v or 'low' in v:
            return 'budget'
        if 'premium' in v or 'high' in v or 'luxury' in v:
            return 'premium'
        return 'mid-range'
 
    @field_validator('key_features', mode='before')
    @classmethod
    def clean_features(cls, v):
        if not isinstance(v, list):
            return []
        return [str(f)[:100] for f in v if f][:5]


class SimulatedReview(BaseModel):
    user_id: str
    asin: str
    reasoning: str = Field(
        max_length=800,
        description="STRICT CONSTRAINT: Under 800 characters."
    )
    rating: int = Field(ge=1, le=5)
    title: str = Field(max_length=200)
    review: str = Field(
        min_length=10,
        description="Full review in the user's voice."
    )
    language: str = "english"

    @field_validator('reasoning', mode='before')
    @classmethod
    def truncate_reasoning(cls, v):
        return truncate(str(v), 800) if v else v

    @field_validator('title', mode='before')
    @classmethod
    def truncate_title(cls, v):
        return truncate(str(v), 200) if v else v

    @field_validator('rating', mode='before')
    @classmethod
    def coerce_and_clamp_rating(cls, v):
        if isinstance(v, str):
            match = re.search(r'\d+', v)
            v = int(match.group()) if match else 3
        return max(1, min(5, int(float(v))))

    @field_validator('review', mode='before')
    @classmethod
    def review_not_placeholder(cls, v):
        placeholders = {'n/a', 'none', 'null', '[review]', 'review text', 'placeholder'}
        if not v or str(v).lower().strip() in placeholders:
            raise ValueError("Review is a placeholder — generation failed")
        return str(v)


class SimulateRequest(BaseModel):
    user_id: Optional[str] = None
    item_asin: str
    item_title: str
    item_description: Optional[str] = ""
    nigerian_mode: bool = True


class SimulateResponse(BaseModel):
    rating: int
    review_title: str
    review_text: str
    persona_type: str
    mode: str
    user_id: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
#  CACHE LAYERS
# ══════════════════════════════════════════════════════════════════

class PersonaCache:
    """Stores Agent 1 dossier profiles on disk."""
    def __init__(self, cache_dir='persona_cache/'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, user_id):
        return os.path.join(
            self.cache_dir,
            f"{hashlib.md5(user_id.encode()).hexdigest()}.json"
        )

    def get(self, user_id):
        path = self._key(user_id)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def set(self, user_id, profile):
        try:
            with open(self._key(user_id), 'w') as f:
                json.dump(profile, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write dossier cache for {user_id}: {e}")

    def exists(self, user_id):
        return os.path.exists(self._key(user_id))

    def size(self):
        try:
            return len(os.listdir(self.cache_dir))
        except Exception:
            return 0


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


    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1500, use_search: bool = False) -> Optional[str]:
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
                result = self._call_provider(key_stat, prompt, temperature, max_tokens, use_search)
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

    def _call_provider(self, key_stat: KeyStats, prompt: str, temperature: float, max_tokens: int, use_search: bool) -> str:
        if key_stat.provider in [
            Provider.GEMINI_20_FLASH,
            Provider.GEMINI_15_FLASH
        ]:
            from google import genai
            from google.genai import types
            client   = genai.Client(api_key=key_stat.api_key)
            if use_search:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            else:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            response = client.models.generate_content(
                model    = key_stat.provider.value,
                contents = prompt,
                config   = config
            )
            return response.text.strip()

        elif key_stat.provider == Provider.CEREBRAS_DEEPSEEK:
            from cerebras.cloud import Cerebras
            client = Cerebras(api_key=key_stat.api_key)
            response = client.chat.completions.create(
                model    = "deepseek-r1-distill-llama-70b",
                messages = [{"role": "user", "content": prompt}],
                max_tokens = min(max_tokens, 1500),
                temperature = temperature
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
                max_tokens = min(max_tokens, 1000),
                temperature = temperature
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
rich_df: Optional[pd.DataFrame] = None
rag_index: Optional[faiss.Index] = None
rag_metadata: Optional[List[dict]] = None
persona_cache: Optional[PersonaCache] = None
embedder: Optional[SentenceTransformer] = None
item_cache_dir = "item_cache/"
key_manager = MultiProviderKeyManager()


# Unified wrapper matching original main.py signature
def generate_with_retry(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1500,
    use_search: bool = False
) -> Optional[str]:
    return key_manager.generate(prompt, temperature, max_tokens, use_search)


# ══════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════

@app.on_event("startup")
def load_artifacts():
    global rich_df, rag_index, rag_metadata, persona_cache, embedder

    # Create directories if they do not exist
    os.makedirs(item_cache_dir, exist_ok=True)
    persona_cache = PersonaCache(cache_dir="persona_cache/")

    # ── Cleaned User DataFrame ───────────────────────────────
    logger.info("Loading cleaned user CSV...")
    if os.path.exists("rich_users.csv"):
        rich_df = pd.read_csv("rich_users.csv")
        logger.info(f"  Cleaned users: {len(rich_df):,}")
    else:
        logger.warning("  rich_users.csv not found!")

    # ── FAISS Vector Index ───────────────────────────────────
    logger.info("Loading RAG FAISS index...")
    if os.path.exists("rag_index.faiss"):
        rag_index = faiss.read_index("rag_index.faiss")
        logger.info(f"  Vectors: {rag_index.ntotal:,}  |  Dim: {rag_index.d}")
    else:
        logger.warning("  rag_index.faiss not found!")

    # ── RAG Metadata Pickle ──────────────────────────────────
    logger.info("Loading RAG metadata pickle...")
    if os.path.exists("rag_meta.pkl"):
        with open("rag_meta.pkl", "rb") as f:
            rag_metadata = pickle.load(f)
        logger.info(f"  Metadata entries: {len(rag_metadata):,}")
    else:
        logger.warning("  rag_meta.pkl not found!")

    # ── Sentence Transformer (CPU Only, Hardcoded) ───────────
    logger.info("Loading embedder (all-MiniLM-L6-v2, device=cpu)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # Assert FAISS index dimension is 384
    if rag_index is not None:
        assert rag_index.d == embedder.get_sentence_embedding_dimension(), (
            f"FATAL: Dimension mismatch — index={rag_index.d}, "
            f"model={embedder.get_sentence_embedding_dimension()}"
        )
        logger.info(f"  Dimension check passed: {rag_index.d}")

    logger.info(f"  Key Manager Pool initialised with {len(key_manager.keys)} provider(s) ✓")
    logger.info("Task A startup complete ✅")



# ══════════════════════════════════════════════════════════════════
#  TOKEN COMPRESSION & COHORT REPRESENTATIVE SELECTION
# ══════════════════════════════════════════════════════════════════

def compress_review(text, max_words=80):
    """Trims review to max_words to save context tokens."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', str(text).strip())
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[.]{2,}', '...', text)
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = ' '.join(words[:max_words])
    last_stop = max(
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?')
    )
    if last_stop > max_words * 3:
        return truncated[:last_stop + 1]
    return truncated + '...'


def select_representative_reviews(texts, ratings, n=6):
    """Picks balanced representative past reviews."""
    if not texts:
        return []
    paired = list(zip(texts, ratings))
    selected = []
    seen_idx = set()

    # 1. lowest rating (complaints)
    min_idx = min(range(len(paired)), key=lambda i: paired[i][1])
    selected.append(paired[min_idx])
    seen_idx.add(min_idx)

    # 2. highest rating (praise voice)
    max_idx = max(range(len(paired)), key=lambda i: paired[i][1])
    if max_idx not in seen_idx:
        selected.append(paired[max_idx])
        seen_idx.add(max_idx)

    # 3. middle rating (nuance)
    mid_idx = min(
        [i for i in range(len(paired)) if i not in seen_idx],
        key=lambda i: abs(paired[i][1] - 3),
        default=None
    )
    if mid_idx is not None:
        selected.append(paired[mid_idx])
        seen_idx.add(mid_idx)

    # 4. latest entries
    for i in range(len(paired) - 1, -1, -1):
        if i not in seen_idx and len(selected) < n:
            selected.append(paired[i])
            seen_idx.add(i)

    return selected


def build_compact_history(texts, ratings, max_words_per_review=80, n_reviews=6):
    """Token-efficient compressed review history block."""
    selected = select_representative_reviews(texts, ratings, n=n_reviews)
    lines = []
    for text, rating in selected:
        compressed = compress_review(text, max_words=max_words_per_review)
        lines.append(f"[{rating}★] {compressed}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  AGENT 1: ANALYST
# ══════════════════════════════════════════════════════════════════

def run_agent1_analyst(
    user_id: str,
    texts: list,
    ratings: list,
) -> PersonaDossier:
    """Builds and caches dossier profiles from a user's review history."""
    cached = persona_cache.get(user_id)
    if cached:
        try:
            return PersonaDossier(**cached)
        except Exception as e:
            logger.warning(f"Cache load failed for {user_id}: {e}")

    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 3.0
    rating_std = round(
        (sum((r - avg_rating) ** 2 for r in ratings) / len(ratings)) ** 0.5, 2
    ) if ratings else 0.0

    history_block = build_compact_history(texts, ratings)
    stats_block = (
        f"Total: {len(ratings)} reviews | "
        f"Avg: {avg_rating}/5 | "
        f"Std: {rating_std} | "
        f"Spread: {sorted(set(ratings))}"
    )

    schema_hint = json.dumps(PersonaDossier.model_json_schema(), indent=2)

    prompt = f"""Analyse this Amazon reviewer and produce their persona dossier.
Discover traits dynamically — do not use generic categories.
Be concise overall. For deep_traits specifically, write each as a full observable phrase — these will be used for keyword matching.
Never exceed the character limits in the schema.
Front-load the most important information.
 
STATS: {stats_block}
 
REVIEWS:
{history_block}
 
Return ONLY valid JSON matching this exact schema:
{schema_hint}
 
Rules:
- deep_traits must be specific observations, not generic single words
- simulation_brief is written directly to the generation model
- never_gives is a list of integers e.g. [5] or []
- No markdown, no explanation, JSON only"""

    for attempt in range(2):
        try:
            raw = generate_with_retry(prompt, temperature=0.3, max_tokens=1500)
            if not raw:
                continue

            raw = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'^```\s*',     '', raw, flags=re.MULTILINE)
            raw = re.sub(r'\s*```$',     '', raw, flags=re.MULTILINE)

            data = json.loads(raw.strip())
            data['user_id'] = user_id
            data['avg_rating'] = avg_rating
            data['rating_std'] = rating_std
            data['review_count'] = len(ratings)

            dossier = PersonaDossier(**data)
            persona_cache.set(user_id, dossier.model_dump())
            return dossier
        except Exception as e:
            logger.warning(f"Agent 1 analyst attempt {attempt+1} failed: {e}")

    # Standard resilient fallback
    logger.error(f"All Agent 1 attempts failed for {user_id} — using standard fallback")
    return PersonaDossier(
        user_id=user_id,
        core_identity="Reviewer with standard and moderate preferences.",
        rating_behaviour=RatingBehaviour(
            tendency=RatingTendency.balanced,
            consistency=Consistency.consistent,
            never_gives=[],
            pattern="Gives balanced reviews based on utility."
        ),
        writing_voice=WritingVoice(
            style=WritingStyle.moderate,
            tone="neutral",
            structure="standard review style.",
            signature_phrases=[]
        ),
        deep_traits=[
            "Writes straightforward reviews of general products.",
            "Values price-to-quality ratio above cosmetics.",
            "Expresses moderate level of scepticism."
        ],
        what_they_care_about=["product quality", "value for money"],
        what_they_ignore=[],
        nigerian_signals=NigerianSignals(
            price_sensitivity=PriceSensitivity.moderate,
            scepticism=SkepticismLevel.moderate,
            community_oriented=False,
            cultural_notes=None
        ),
        simulation_brief=(
            f"Write a balanced, neutral review consistent with a {avg_rating:.1f}-star average."
        ),
        avg_rating=avg_rating,
        rating_std=rating_std,
        review_count=len(ratings)
    )


# ══════════════════════════════════════════════════════════════════
#  GOOGLE SEARCH-GROUNDED PRODUCT CARD ENRICHMENT
# ══════════════════════════════════════════════════════════════════

def enrich_item_card(
    asin: str,
    title: str,
    description: str,
    category: str,
) -> ItemCard:
    """Enriches product description using Gemini search grounding."""
    cache_path = os.path.join(item_cache_dir, f"{asin}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return ItemCard(**json.load(f))
        except Exception:
            pass

    prompt = f"""Search for this product and return a structured description.
 
Product title    : {title}
Category         : {category}
Existing description (may be empty): {description[:200] if description else ''}
 
Return a JSON object with exactly these fields:
{{
  "product_summary"  : "<2-3 sentence plain English description of what this product is>",
  "key_features"     : ["<feature 1>", "<feature 2>", "<feature 3>"],
  "typical_use_case" : "<who uses this and why, one sentence>",
  "price_tier"       : "<exactly one of: budget | mid-range | premium>",
  "brand_notes"      : "<one sentence about the brand, or empty string>"
}}"""

    item_card = None
    try:
        raw = generate_with_retry(prompt, temperature=0.1, max_tokens=1000, use_search=True)
        if raw:
            raw = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'^```\s*',     '', raw, flags=re.MULTILINE)
            raw = re.sub(r'\s*```$',     '', raw, flags=re.MULTILINE)

            data = json.loads(raw.strip())
            item_card = ItemCard(**data)
    except Exception as e:
        logger.warning(f"ItemCard enrichment failed for {asin} ({e}) — using metadata fallback")

    if item_card is None:
        item_card = ItemCard(
            product_summary=description[:300] if description else f"{category} product: {title}",
            key_features=[],
            typical_use_case=category,
            price_tier="mid-range",
            brand_notes=""
        )

    item_card.asin = asin
    item_card.title = title

    try:
        with open(cache_path, 'w') as f:
            json.dump(item_card.model_dump(), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to cache ItemCard for {asin}: {e}")

    return item_card


# ══════════════════════════════════════════════════════════════════
#  RAG RETRIEVAL FROM RAG DATASETS
# ══════════════════════════════════════════════════════════════════

def retrieve_for_agent2(
    user_id: str,
    item_title: str,
    item_description: str,
    item_category: str,
    top_k: int = 4,
    same_category_boost: bool = True
) -> List[dict]:
    """Retrieves similar reviews by this user from the FAISS RAG index."""
    if rag_index is None or not rag_metadata:
        logger.warning("RAG index/metadata not loaded!")
        return []

    # Cosine search query
    query_parts = [p for p in [item_title, item_description, item_category] if p]
    query = ' '.join(query_parts)[:512]

    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    search_k = rag_index.ntotal
    scores, indices = rag_index.search(query_embedding, search_k)
    scores = scores[0]
    indices = indices[0]

    user_results = []
    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(rag_metadata):
            continue

        record = rag_metadata[idx]
        if record['user_id'] != user_id:
            continue

        if len(record['text'].split()) < 10:
            continue

        user_results.append({
            'text': record['text'],
            'rating': record['rating'],
            'category': record['category'],
            'asin': record['asin'],
            'timestamp': record.get('timestamp', 0),
            'similarity_score': float(score),
            'same_category': (
                record['category'].lower() == item_category.lower()
                if item_category else False
            ),
            'source': 'rag'
        })

    # Boost category matches
    if same_category_boost:
        for r in user_results:
            if r['same_category']:
                r['similarity_score'] += 0.15

    # Rank
    user_results.sort(key=lambda x: x['similarity_score'], reverse=True)

    # Check same category matches
    same_cat_found = any(r['same_category'] for r in user_results[:top_k])

    if user_results and same_cat_found:
        return user_results[:top_k]

    if user_results and not same_cat_found:
        logger.info(
            f"No same-category results for {user_id} in '{item_category}' — "
            f"returning best semantic matches"
        )
        return user_results[:top_k]

    # Fallback to latest reviews
    logger.warning(f"No RAG results for {user_id} — falling back to latest reviews")

    all_user_records = [
        {
            'text': m['text'],
            'rating': m['rating'],
            'category': m['category'],
            'asin': m['asin'],
            'timestamp': m.get('timestamp', 0),
            'similarity_score': 0.0,
            'same_category': (
                m['category'].lower() == item_category.lower()
                if item_category else False
            ),
            'source': 'fallback_latest'
        }
        for m in rag_metadata
        if m['user_id'] == user_id
        and len(m['text'].split()) >= 10
    ]

    if not all_user_records:
        return []

    all_user_records.sort(key=lambda x: x['timestamp'], reverse=True)
    return all_user_records[:top_k]


def format_rag_examples(retrieved_reviews: List[dict]) -> str:
    """Formats list of retrieved reviews into clean prompt examples."""
    if not retrieved_reviews:
        return "No similar past reviews found for this user."

    lines = []
    for i, review in enumerate(retrieved_reviews):
        category_note = " [same category]" if review.get('same_category') else ""
        source_note = " [fallback — latest reviews]" if review.get('source') == 'fallback_latest' else ""

        lines.append(
            f"Past review {i+1}{category_note}{source_note}\n"
            f"  Rating : {int(review['rating'])}★\n"
            f"  Text   : \"{review['text'].strip()}\""
        )
    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  AGENT 2: RAG REVIEW GENERATION
# ══════════════════════════════════════════════════════════════════

def run_agent2_with_rag(
    dossier: PersonaDossier,
    item_asin: str,
    item_title: str,
    item_description: str,
    item_category: str,
    nigerian_language: Optional[str] = None,
    top_k: int = 4,
    item_card: Optional[ItemCard] = None
) -> SimulatedReview:
    """Generates the review text matching a specific user dossier and rating distribution."""
    # 1. RAG retrieval
    retrieved = retrieve_for_agent2(
        user_id=dossier.user_id,
        item_title=item_title,
        item_description=item_description,
        item_category=item_category,
        top_k=top_k
    )
    rag_block = format_rag_examples(retrieved)

    # 2. Product context card
    if item_card is not None:
        product_block = f"""Title         : {item_card.title}
ASIN          : {item_asin}
Category      : {item_category}
What it is    : {item_card.product_summary}
Key features  : {' | '.join(item_card.key_features) or 'not available'}
Typical use   : {item_card.typical_use_case}
Price tier    : {item_card.price_tier}
Brand notes   : {item_card.brand_notes or 'none'}"""
    else:
        product_block = f"""Title      : {item_title}
ASIN       : {item_asin}
Category   : {item_category}
Description: {str(item_description)[:400]}"""

    # Bounding window rating calculation
    user_mean = dossier.avg_rating
    user_std = dossier.rating_std if dossier.rating_std > 0 else 0.5
    
    lower_bound = max(1, round(user_mean - (2 * user_std)))
    upper_bound = min(5, round(user_mean + (2 * user_std)))

    # Cultural language instructions
    lang_instructions = {
        'pidgin': "Nigerian Pidgin — weave in naturally: 'e good o', 'I no go lie', 'wahala', 'sha'",
        'yoruba_influenced': "Yoruba-influenced English — 'omo', 'my people', occasional Yoruba words",
        'igbo_influenced': "Igbo-influenced English — 'nna', 'chai', direct assertive tone",
        'hausa_influenced': "Hausa-influenced English — 'wallahi', 'alhamdulillah', respectful tone",
    }
    lang_line = ""
    if nigerian_language and nigerian_language in lang_instructions:
        lang_line = f"\nLANGUAGE STYLE: {lang_instructions[nigerian_language]}\n"

    # Stage 1: Estimating Rating with Front-Loaded Reasoning
    committed_rating = round(user_mean)
    committed_reason = "Fallback — calculation window defaulted to historical mean."

    rating_prompt = f"""You are analyzing a product to determine what star rating this unique reviewer profile would assign.
    
CRITICAL PROFILE CONSTRAINTS:
- Reviewer baseline average rating: {user_mean:.2f} / 5 stars
- Reviewer baseline variance deviation: {user_std:.2f}
- STALWART STATISTICAL LIMITS: Your final choice MUST stay strictly within {lower_bound} to {upper_bound} stars based on their lifetime history.
- Never gives options: {dossier.rating_behaviour.never_gives or 'none'}

PRODUCT CONTEXT:
{product_block}
 
PAST USER EXPERIENCES FOR COMPARISON:
{rag_block}
 
OUTPUT DIRECTIONS:
Evaluate how the product highlights map to what this persona cares about. 
Write your analytical justification FIRST, then conclude with the numerical rating token.

Return ONLY a raw JSON object structured exactly like this:
{{
  "reasoning": "<write your contextual behavior analysis here under 200 characters>",
  "rating": <provide a single integer within the user's hard bounds of {lower_bound} to {upper_bound}>
}}
No markdown formatting fences. No alternative fields."""

    try:
        raw1 = generate_with_retry(rating_prompt, temperature=0.1, max_tokens=250)
        if raw1:
            raw1 = re.sub(r'^```json\s*', '', raw1, flags=re.MULTILINE)
            raw1 = re.sub(r'^```\s*',     '', raw1, flags=re.MULTILINE)
            raw1 = re.sub(r'\s*```$',     '', raw1, flags=re.MULTILINE)

            d1 = json.loads(raw1.strip())
            parsed_rating = int(float(d1.get('rating', committed_rating)))
            committed_rating = max(lower_bound, min(upper_bound, parsed_rating))
            committed_reason = str(d1.get('reasoning', committed_reason))[:200]
    except Exception as e:
        logger.warning(f"Rating estimation pipeline failed ({e}) — defaulting to safe bounds.")
        committed_rating = max(lower_bound, min(upper_bound, round(user_mean)))

    # Stage 2: Simulating Review Conditional on Rating
    output_example = (
        '{\n'
        '  "reasoning": "' + committed_reason.replace('"', '\\"') + '",\n'
        '  "rating": ' + str(committed_rating) + ',\n'
        '  "title": "Enter short title here",\n'
        '  "review": "Enter generated body review text here matching the tone instructions.",\n'
        '  "user_id": "' + dossier.user_id + '",\n'
        '  "asin": "' + item_asin + '",\n'
        '  "language": "english"\n'
        '}'
    )

    review_prompt = f"""You are generating the final text review matching a specific consumer's voice.
The final rating is permanently locked at exactly: {committed_rating}★
Your generated text body must be completely coherent with a {committed_rating}★ evaluation.

REVIEWER VOICE PARAMETERS:
{dossier.simulation_brief}
Identity Profile: {dossier.core_identity}
Tone & Cadence   : {dossier.writing_voice.tone} | {dossier.writing_voice.style}
Signature Phrases: {', '.join(dossier.writing_voice.signature_phrases) or 'none found'}

PAST STRUCTURAL TEXT EXAMPLES:
{rag_block}
{lang_line}
PRODUCT METADATA CARD:
{product_block}

ANTI-DRIFT RUNTIME INSTRUCTIONS:
1. Mirror the historical length pattern shown in the past structural text examples. Do not write filler.
2. Weave the following traits naturally into the narrative without referencing them directly:
{chr(10).join(f'   - {t}' for t in dossier.deep_traits)}
3. Embody the exact level of critical nuance required to justify a score of {committed_rating}★.

OUTPUT RULES:
- Return ONLY a single JSON object.
- The "rating" key must appear before the "review" body key.
- No markdown fences or formatting characters outside the json structure.

MATCH THIS TARGET STRUCTURAL FORMAT:
{output_example}
"""

    for attempt in range(2):
        try:
            raw = generate_with_retry(review_prompt, temperature=0.6, max_tokens=800)
            if raw:
                raw = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
                raw = re.sub(r'^```\s*',     '', raw, flags=re.MULTILINE)
                raw = re.sub(r'\s*```$',     '', raw, flags=re.MULTILINE)
                raw = raw.strip()

                brace_count = 0
                end_pos = 0
                for i, char in enumerate(raw):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    raw = raw[:end_pos]

                data = json.loads(raw)
                data['user_id'] = dossier.user_id
                data['asin'] = item_asin
                data['language'] = nigerian_language or 'english'
                data['rating'] = committed_rating
                data['reasoning'] = committed_reason

                return SimulatedReview(**data)
        except Exception as e:
            logger.warning(f"Agent 2 review layout failed on attempt {attempt+1}: {e}")

    # Ultimate recoverability fallback
    return SimulatedReview(
        user_id=dossier.user_id,
        asin=item_asin,
        reasoning=committed_reason,
        rating=committed_rating,
        title=f"Okay: {item_title[:30]}",
        review="E good o! I bought this recently and it works very well. Quite solid value for money, I would advise you try it.",
        language=nigerian_language or 'english'
    )


# ══════════════════════════════════════════════════════════════════
#  CATEGORY FINDER HELPERS
# ══════════════════════════════════════════════════════════════════

def determine_category(asin: str, title: str) -> str:
    """Helper to detect product category dynamically."""
    if rich_df is not None and asin in rich_df['parent_asin'].values:
        cat = rich_df[rich_df['parent_asin'] == asin]['main_category'].iloc[0]
        if pd.notna(cat) and str(cat).strip():
            return str(cat).strip()

    if rag_metadata:
        for r in rag_metadata:
            if r.get('asin') == asin:
                cat = r.get('category')
                if cat:
                    return cat

    t = title.lower()
    if any(w in t for w in ["game", "xbox", "playstation", "ps4", "ps5", "nintendo", "switch", "controller", "gaming", "console"]):
        return "Video Games and Software"
    return "Beauty"


# ══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "users_in_dataset": len(rich_df["user_id"].unique()) if rich_df is not None else 0,
        "rag_vectors": rag_index.ntotal if rag_index is not None else 0,
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



@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    """
    Simulates a review for a user on a given item.
    - Uses cached dossier if available, otherwise analyst Agent 1 builds it.
    - Grounding Search-enrichment on product cards.
    - Category-aware RAG vector search retrieval (FAISS).
    - Creativity Agent 2 generation conditioned on clamped rating distribution.
    - Synthetic persona fallback for cold-start (unrecognized user IDs or None).
    """
    user_id = req.user_id
    asin = req.item_asin
    title = req.item_title
    description = req.item_description or ""
    nigerian_mode = req.nigerian_mode

    category = determine_category(asin, title)
    nigerian_language = "pidgin" if nigerian_mode else None

    # Check if user exists in rich_df review history
    user_exists = False
    user_reviews = pd.DataFrame()
    if user_id and rich_df is not None:
        user_reviews = rich_df[rich_df['user_id'] == user_id]
        if not user_reviews.empty:
            user_exists = True

    # ── DETERMINE MODE & CHOOSE PROFILE DOSSIER ─────────────────
    if user_id and user_exists:
        mode = "history_based"
        logger.info(f"Running pipeline in history-based mode for user {user_id}")
        texts = user_reviews['text'].tolist()
        ratings = user_reviews['rating'].tolist()

        # Agent 1 (Analyst) dossier extraction
        dossier = run_agent1_analyst(user_id=user_id, texts=texts, ratings=ratings)
    else:
        mode = "cold_start"
        logger.info("Running pipeline in cold-start mode with synthetic persona dossier")
        
        # Conforms strictly to the dossier schemas
        dossier = PersonaDossier(
            user_id="cold_start",
            core_identity="A typical Nigerian consumer who values price, effectiveness, and reliability.",
            rating_behaviour=RatingBehaviour(
                tendency=RatingTendency.balanced,
                consistency=Consistency.consistent,
                never_gives=[],
                pattern="Gives constructive ratings based on price-to-quality ratio."
            ),
            writing_voice=WritingVoice(
                style=WritingStyle.moderate,
                tone="friendly and value-conscious",
                structure="Starts with direct utility, references the price, then lists pros/cons.",
                signature_phrases=["e good o", "value for money"]
            ),
            deep_traits=[
                "Seeks robust value for money; very cost-conscious.",
                "Sceptical of cosmetics claims; values utility.",
                "Prefers buying products with community backing."
            ],
            what_they_care_about=["price consciousness", "effectiveness", "community recommendations"],
            what_they_ignore=[],
            context_clues="Value-driven consumer based in a communal society.",
            nigerian_signals=NigerianSignals(
                price_sensitivity=PriceSensitivity.high,
                scepticism=SkepticismLevel.moderate,
                community_oriented=True,
                cultural_notes="Focuses on value for money and community references."
            ),
            simulation_brief="Embody a value-conscious Nigerian consumer. Evaluate utility vs price.",
            avg_rating=4.0,
            rating_std=0.5,
            review_count=1
        )

    # ── ITEM ENRICHMENT ──────────────────────────────────────────
    logger.info(f"Enriching product card details for {asin}")
    item_card = enrich_item_card(
        asin=asin,
        title=title,
        description=description,
        category=category
    )

    # ── AGENT 2 WITH RAG SIMULATION ──────────────────────────────
    logger.info("Running Agent 2 generator with RAG and profile constraints")
    review = run_agent2_with_rag(
        dossier=dossier,
        item_asin=asin,
        item_title=title,
        item_description=description,
        item_category=category,
        nigerian_language=nigerian_language,
        top_k=4,
        item_card=item_card
    )

    return SimulateResponse(
        rating=review.rating,
        review_title=review.title,
        review_text=review.review,
        persona_type=dossier.rating_behaviour.tendency.value.capitalize(),
        mode=mode,
        user_id=user_id if user_id else "cold_start"
    )
