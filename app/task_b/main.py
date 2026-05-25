"""
Task B - Recommendation API.

Runs FAISS retrieval over item artifacts, then uses an LLM reranker with
explicit Nigerian language conditioning.
"""

import os
import pickle
import random
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import faiss
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)

app = FastAPI(title="Task B - Recommendation API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


LANGUAGE_PROMPTS = {
    "pidgin": {
        "name": "Nigerian Pidgin",
        "system": """You are recommending products to a Nigerian user.
Respond ENTIRELY in Nigerian Pidgin English mixed with English.
Your explanations MUST contain Pidgin phrases like:
"e good o", "e no go fail you", "I no go lie",
"make you try am", "e dey work well-well",
"value for money no be here", "e go serve you well",
"my people dem like am", "you go enjoy am"
Sound like a trusted Nigerian friend, not a formal AI.
EVERY explanation must have at least one Pidgin phrase.""",
    },
    "yoruba": {
        "name": "Yoruba-English Mix",
        "system": """You are recommending products to a Yoruba-speaking Nigerian.
Mix Yoruba words and phrases naturally into English explanations.
MUST include phrases like:
"o dara gan-an" (very good), "E gbiyanju" (please try it),
"o to iye re" (worth the price), "awon eniyan feran" (people love it),
"ko ni dara ju" (it's quite good), "E kaabo" (welcome)
Every recommendation must feel authentically Yoruba-Nigerian.""",
    },
    "hausa": {
        "name": "Hausa-English Mix",
        "system": """You are recommending products to a Hausa-speaking Nigerian.
Mix Hausa words naturally into English explanations.
MUST include phrases like:
"yana da kyau sosai" (very good), "mai rahusa" (affordable),
"ina ba da shawara" (I recommend), "mutane suna so" (people like it),
"ya yi aiki sosai" (works very well), "Sannu" (hello/greetings)
Every recommendation must feel authentically Hausa-Nigerian.""",
    },
    "igbo": {
        "name": "Igbo-English Mix",
        "system": """You are recommending products to an Igbo-speaking Nigerian.
Mix Igbo words naturally into English explanations.
MUST include phrases like:
"di mma nke oma" (very good), "o di ire ire" (good value),
"I ga-amasi ya" (you will like it), "o na-aru oru" (it works),
"nwanne" (brother/sister), "Nnoo" (welcome)
Every recommendation must feel authentically Igbo-Nigerian.""",
    },
    "english": {
        "name": "Nigerian English",
        "system": """You are recommending products to a Nigerian user.
Use Nigerian English - practical, value-conscious, communal.
Reference Nigerian consumer concerns: price sensitivity,
value for money, family use, trusted brands, effectiveness.
Sound warm and helpful like a knowledgeable Nigerian friend.""",
    },
}


class Provider(Enum):
    GEMINI_20 = "gemini-2.0-flash"
    GEMINI_15 = "gemini-1.5-flash"
    CEREBRAS = "cerebras-llama-3.3-70b"
    DEEPSEEK = "deepseek-chat"
    GROQ = "groq-llama-3.3-70b"


@dataclass
class KeyStats:
    provider: Provider
    api_key: str
    requests_made: int = 0
    errors_429: int = 0
    errors_other: int = 0
    total_latency: float = 0.0
    is_exhausted: bool = False

    @property
    def avg_latency(self):
        return self.total_latency / max(self.requests_made, 1)

    def to_dict(self):
        return {
            "provider": self.provider.value,
            "requests_made": self.requests_made,
            "errors_429": self.errors_429,
            "avg_latency_s": round(self.avg_latency, 3),
            "error_rate_pct": round(
                (self.errors_429 + self.errors_other)
                / max(self.requests_made, 1)
                * 100,
                1,
            ),
            "status": "exhausted" if self.is_exhausted else "active",
        }


class MultiProviderKeyManager:
    def __init__(self):
        self.keys: list[KeyStats] = []
        configs = [
            (Provider.GEMINI_20, "GOOGLE_API_KEY_1"),
            (Provider.GEMINI_15, "GOOGLE_API_KEY_2"),
            (Provider.CEREBRAS, "CEREBRAS_API_KEY"),
            (Provider.DEEPSEEK, "DEEPSEEK_API_KEY"),
            (Provider.GROQ, "GROQ_API_KEY"),
        ]
        for provider, env_var in configs:
            key = os.environ.get(env_var, "")
            if key and not key.startswith("your_"):
                self.keys.append(KeyStats(provider, key))
                print(f"{provider.value} loaded")
            else:
                print(f"{env_var} not set")

        if not self.keys:
            raise ValueError("No API keys configured")

    def generate(self, prompt: str) -> Optional[str]:
        for ks in self.keys:
            if ks.is_exhausted:
                continue
            start = time.time()
            try:
                result = self._call(ks, prompt)
                elapsed = time.time() - start
                ks.requests_made += 1
                ks.total_latency += elapsed
                print(f"{ks.provider.value} responded in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start
                ks.total_latency += elapsed
                ks.requests_made += 1
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    ks.errors_429 += 1
                    if "quota" in err.lower():
                        ks.is_exhausted = True
                        print(f"{ks.provider.value} quota exhausted")
                    else:
                        print(f"{ks.provider.value} rate limited, trying next")
                        time.sleep(5)
                else:
                    ks.errors_other += 1
                    print(f"{ks.provider.value} error: {err[:80]}")
        return None

    def _call(self, ks: KeyStats, prompt: str) -> str:
        p = ks.provider
        if p in [Provider.GEMINI_20, Provider.GEMINI_15]:
            from google import genai

            client = genai.Client(api_key=ks.api_key)
            models_to_try = [p.value]
            if p == Provider.GEMINI_15:
                models_to_try.extend(["gemini-2.0-flash", "gemini-1.5-flash-latest"])
            else:
                models_to_try.extend(["gemini-2.0-flash-exp"])

            last_err = None
            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(model=model_name, contents=prompt)
                    return response.text.strip()
                except Exception as e:
                    last_err = e
                    print(f"⚠️ Gemini model {model_name} failed: {str(e)[:80]}")
            raise last_err

        if p == Provider.CEREBRAS:
            try:
                from cerebras.cloud.sdk import Cerebras
            except Exception:
                from cerebras.cloud import Cerebras

            client = Cerebras(api_key=ks.api_key)
            models_to_try = ["llama3.3-70b", "llama-3.3-70b", "deepseek-r1-distill-llama-70b"]
            last_err = None
            for model_name in models_to_try:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000,
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    last_err = e
                    print(f"⚠️ Cerebras model {model_name} failed: {str(e)[:80]}")
            raise last_err

        if p == Provider.DEEPSEEK:
            from openai import OpenAI

            client = OpenAI(api_key=ks.api_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        if p == Provider.GROQ:
            from groq import Groq

            client = Groq(api_key=ks.api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        raise ValueError(f"Unsupported provider: {p}")


key_manager = MultiProviderKeyManager()


def generate(prompt: str) -> Optional[str]:
    return key_manager.generate(prompt)


class RecommendRequest(BaseModel):
    user_id: Optional[str] = None
    product_type: Optional[str] = None
    priority: Optional[str] = None
    avoid: Optional[str] = ""
    language: str = "pidgin"
    nigerian_mode: bool = True


class RecommendationItem(BaseModel):
    rank: int
    asin: str
    title: str
    avg_rating: float
    review_count: int
    explanation: str


class RecommendResponse(BaseModel):
    reasoning: str
    recommendations: list[RecommendationItem]
    persona_type: str
    language: str


index: Optional[faiss.Index] = None
item_meta: Optional[pd.DataFrame] = None
embedder: Optional[SentenceTransformer] = None
pop_lookup: dict[str, float] = {}


@app.on_event("startup")
def startup():
    global index, item_meta, embedder, pop_lookup

    index_path = APP_DIR / "items.index"
    meta_path = APP_DIR / "item_meta.pkl"
    if not index_path.exists():
        raise RuntimeError(f"MISSING ARTIFACT: {index_path.name}")
    if not meta_path.exists():
        raise RuntimeError(f"MISSING ARTIFACT: {meta_path.name}")

    index = faiss.read_index(str(index_path))
    with meta_path.open("rb") as f:
        item_meta = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    assert index.d == 384, f"Wrong FAISS dimension: {index.d}"
    assert embedder.get_sentence_embedding_dimension() == 384

    if "pop_score" not in item_meta.columns:
        max_reviews = item_meta["review_count"].quantile(0.95)
        item_meta["pop_score"] = (
            item_meta["review_count"].clip(upper=max_reviews) / max_reviews
        ) * item_meta["avg_rating"] / 5.0

    pop_lookup = dict(zip(item_meta["parent_asin"], item_meta["pop_score"]))
    print(f"Task B ready: {index.ntotal} vectors, {len(item_meta)} metadata rows")


def retrieve_items(query_text: str, n_results: int = 20) -> list[dict]:
    if index is None or item_meta is None or embedder is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    query_vec = embedder.encode([query_text], normalize_embeddings=True).astype(
        "float32"
    )
    scores, indices = index.search(query_vec, n_results * 2)

    items = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = item_meta.iloc[idx]
        if meta["review_count"] < 5:
            continue
        if len(str(meta["title"])) < 10:
            continue

        sem_score = float(score)
        pop_score = float(pop_lookup.get(meta["parent_asin"], 0.0))
        items.append(
            {
                "asin": str(meta["parent_asin"]),
                "title": str(meta["title"])[:80],
                "avg_rating": round(float(meta["avg_rating"]), 1),
                "review_count": int(meta["review_count"]),
                "sem_score": sem_score,
                "pop_score": pop_score,
                "final_score": 0.5 * sem_score + 0.5 * pop_score,
            }
        )

    items.sort(key=lambda x: x["final_score"], reverse=True)
    return items[:n_results]


def post_filter(items: list[dict], avoid: str) -> list[dict]:
    if not avoid or avoid.strip() == "":
        return items
    avoid_lower = avoid.lower().strip()
    return [item for item in items if avoid_lower not in item["title"].lower()]


def language_fallback(language: str, fill: bool = False) -> str:
    if fill:
        fallbacks = {
            "pidgin": "E dey work well-well, make you try am!",
            "yoruba": "O dara, E gbiyanju!",
            "hausa": "Yana da kyau, gwada shi!",
            "igbo": "O di mma, nwanne!",
            "english": "Good match for your preferences.",
        }
    else:
        fallbacks = {
            "pidgin": "E good o! This one go work well for you.",
            "yoruba": "O dara gan-an! E gbiyanju.",
            "hausa": "Yana da kyau! Ina ba da shawara.",
            "igbo": "O di mma! I ga-amasi ya.",
            "english": "Great match for your needs and preferences.",
        }
    return fallbacks.get(language, fallbacks["english"])


def llm_rerank(
    persona_query: str,
    candidates: list[dict],
    language: str = "pidgin",
    nigerian_mode: bool = True,
) -> dict:
    if not candidates:
        return {"reasoning": "", "reranked": []}

    language = (language or "english").lower()
    shuffled = candidates.copy()
    random.shuffle(shuffled)

    lang_config = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["english"])
    language_block = lang_config["system"] if nigerian_mode else ""
    candidate_list = "\n".join(
        [
            f"{i + 1}. {item['title']} "
            f"(star {item['avg_rating']}, {item['review_count']} reviews)"
            for i, item in enumerate(shuffled)
        ]
    )

    prompt = f"""{language_block}

USER PERSONA:
{persona_query}

CANDIDATE PRODUCTS TO RANK:
{candidate_list}

INSTRUCTIONS:
- Rerank these products from most to least relevant for this user
- Write your reasoning and each explanation in {lang_config['name']}
- Each explanation MUST use language-appropriate phrases
- Be specific about WHY each product fits this user
- Do NOT use generic phrases like "Strong match" or "Good product"

Output in EXACTLY this format:

REASONING: [2-3 sentences in {lang_config['name']} about this user's needs]
RANKING:
1. [exact product title] | [specific explanation in {lang_config['name']}]
2. [exact product title] | [specific explanation in {lang_config['name']}]
3. [exact product title] | [specific explanation in {lang_config['name']}]
4. [exact product title] | [specific explanation in {lang_config['name']}]
5. [exact product title] | [specific explanation in {lang_config['name']}]"""

    raw = generate(prompt)

    if not raw or "RANKING:" not in raw:
        fallback_text = language_fallback(language)
        return {
            "reasoning": "Top picks based on your preferences.",
            "reranked": [
                {**item, "explanation": fallback_text} for item in candidates[:5]
            ],
        }

    reasoning_match = re.search(r"REASONING:\s*(.*?)(?=RANKING:)", raw, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    ranking_matches = re.findall(r"\d+\.\s+(.+?)\s*\|\s*(.+)", raw)

    reranked = []
    used = set()
    for llm_title, explanation in ranking_matches:
        best_match = None
        best_score = 0
        for i, candidate in enumerate(shuffled):
            if i in used:
                continue
            llm_words = set(llm_title.lower().split())
            real_words = set(candidate["title"].lower().split())
            overlap = len(llm_words & real_words)
            if overlap > best_score:
                best_score = overlap
                best_match = (i, candidate)

        if best_match and best_score > 0:
            used.add(best_match[0])
            reranked.append({**best_match[1], "explanation": explanation.strip()})

    if len(reranked) < 5:
        for i, candidate in enumerate(shuffled):
            if i not in used and len(reranked) < 5:
                reranked.append(
                    {
                        **candidate,
                        "explanation": language_fallback(language, fill=True),
                    }
                )
                used.add(i)

    return {"reasoning": reasoning, "reranked": reranked}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    product = req.product_type or "beauty personal care"
    priority = req.priority or "quality and effectiveness"
    language = (req.language or "pidgin").lower()

    query = f"gentle {product} product. {priority}. natural effective quality."
    persona_string = (
        f"Nigerian user looking for {product} products. "
        f"Prioritises {priority}. "
        f"Avoids: {req.avoid or 'nothing specific'}."
    )

    candidates = retrieve_items(query, n_results=20)
    filtered = post_filter(candidates, req.avoid or "")[:10]
    if not filtered:
        filtered = candidates[:10]

    result = llm_rerank(
        persona_query=persona_string,
        candidates=filtered,
        language=language,
        nigerian_mode=req.nigerian_mode,
    )

    recommendations = [
        RecommendationItem(
            rank=i + 1,
            asin=item["asin"],
            title=item["title"],
            avg_rating=item["avg_rating"],
            review_count=item["review_count"],
            explanation=item.get("explanation", ""),
        )
        for i, item in enumerate(result["reranked"][:5])
    ]

    return RecommendResponse(
        reasoning=result["reasoning"],
        recommendations=recommendations,
        persona_type="cold_start",
        language=language,
    )


@app.get("/products")
def get_products(search: str = "", limit: int = 100):
    if item_meta is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    filtered = item_meta[
        (item_meta["title"].notna())
        & (item_meta["title"].str.len() > 15)
        & (item_meta["title"].str.len() < 100)
        & (item_meta["review_count"] >= 10)
        & (
            ~item_meta["title"]
            .str.lower()
            .str.match(
                r"^(great|good|love|nice|works|this|i |the |very|best|ok)",
                na=False,
            )
        )
    ].copy()

    if search:
        filtered = filtered[filtered["title"].str.contains(search, case=False, na=False)]

    top = filtered.nlargest(limit, "review_count")
    return {
        "products": [
            {
                "asin": row["parent_asin"],
                "title": row["title"][:80],
                "avg_rating": round(float(row["avg_rating"]), 1),
                "review_count": int(row["review_count"]),
            }
            for _, row in top.iterrows()
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "items_indexed": index.ntotal if index else 0,
        "dimension": index.d if index else 0,
    }


@app.get("/")
def root():
    return {
        "name": "DSN x BCT - Task B Recommendation API",
        "team": "Drizzy x Metro",
        "version": "1.0.0",
        "endpoints": [
            "POST /recommend",
            "GET /products",
            "GET /health",
            "GET /stats",
            "GET /docs",
        ],
        "metrics": {
            "ndcg_at_10": 0.4210,
            "hit_rate_10": 0.6600,
            "languages": ["pidgin", "yoruba", "hausa", "igbo", "english"],
            "user_modes": ["cold_start", "one_shot", "history_based"],
        },
    }


@app.get("/stats")
def stats():
    return {
        "providers": [k.to_dict() for k in key_manager.keys],
        "total_requests": sum(k.requests_made for k in key_manager.keys),
        "active_provider": next(
            (k.provider.value for k in key_manager.keys if not k.is_exhausted),
            "none",
        ),
    }
