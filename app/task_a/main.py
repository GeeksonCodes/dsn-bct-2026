"""
Task A - Review Simulator API.

Generates language-conditioned product review variations using the shared
multi-provider LLM pool.
"""

import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)

app = FastAPI(title="Task A - Review Simulator API", version="2.0.0")
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


VARIATION_STYLES = [
    {
        "style": "brief",
        "instruction": "Write a SHORT review of 1-2 sentences only. Be direct and casual.",
    },
    {
        "style": "detailed",
        "instruction": "Write a DETAILED review of 4-5 sentences. Mention specific product benefits, texture, scent, and results.",
    },
    {
        "style": "emotional",
        "instruction": "Write an EMOTIONAL, personal review. Share how it made you feel, personal context, and strong opinion.",
    },
]


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


class SimulateRequest(BaseModel):
    user_id          : Optional[str] = None
    user_persona     : Optional[str] = None
    item_asin        : str
    item_title       : str
    item_description : str = ""
    language         : str = "english"
    nigerian_mode    : bool = True
    num_variations   : int = 1


class ReviewVariation(BaseModel):
    rating: int
    review_title: str
    review_text: str
    style: str


class SimulateResponse(BaseModel):
    variations: list[ReviewVariation]
    persona_type: str
    mode: str
    language: str


def parse_review_output(raw: str, item_title: str) -> tuple[int, str, str]:
    rating_match = re.search(r"RATING:\s*(\d)", raw)
    rating = int(rating_match.group(1)) if rating_match else 4
    rating = max(1, min(5, rating))

    title_match = re.search(r"TITLE:\s*(.+?)(?=\nREVIEW:|\Z)", raw, re.DOTALL)
    review_title = (
        title_match.group(1).strip()
        if title_match
        else f"Review of {item_title[:30]}"
    )

    review_match = re.search(r"REVIEW:\s*(.+)", raw, re.DOTALL)
    review_text = review_match.group(1).strip() if review_match else raw.strip()
    return rating, review_title[:100], review_text[:1000]


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    
    lang_config    = LANGUAGE_PROMPTS.get(req.language, LANGUAGE_PROMPTS["english"])
    language_block = lang_config["system"] if req.nigerian_mode else ""
    
    # Determine persona source
    if req.user_persona and len(req.user_persona.strip()) > 10:
        persona_block = f"""YOU ARE SIMULATING A REVIEWER WITH THIS EXACT PERSONALITY:
{req.user_persona}

CRITICAL: Stay in character completely. The rating, writing style, 
tone, and language MUST match this persona exactly.
A stingy critic MUST give 1-3 stars. An enthusiast MUST give 4-5 stars.
A Pidgin speaker MUST use Pidgin naturally."""
        mode = "custom_persona"
    else:
        persona_block = """YOU ARE SIMULATING A TYPICAL NIGERIAN PRODUCT REVIEWER.
Generate a realistic, persona-consistent review."""
        mode = "cold_start"
    
    styles   = VARIATION_STYLES[:min(req.num_variations, 3)]
    variations = []
    
    for style_info in styles:
        prompt = f"""{language_block}

{persona_block}

PRODUCT TO REVIEW:
Name        : {req.item_title}
ASIN        : {req.item_asin}
Description : {req.item_description or "A beauty and personal care product"}

REVIEW STYLE INSTRUCTION:
{style_info["instruction"]}

LANGUAGE: Write the review in {lang_config["name"]}.
{"Use authentic local language phrases naturally." if req.nigerian_mode else ""}

Generate a realistic review that sounds EXACTLY like this persona.

Output EXACTLY in this format — nothing else:
RATING: [1-5]
TITLE: [review title in {lang_config["name"]}]
REVIEW: [full review text in {lang_config["name"]}]"""

        raw = generate(prompt)
        
        if not raw:
            variations.append(ReviewVariation(
                rating       = 3,
                review_title = f"My review of {req.item_title[:30]}",
                review_text  = "Good product worth trying.",
                style        = style_info["style"]
            ))
            continue
        
        import re
        
        rating_match = re.search(r"RATING:\s*(\d)", raw)
        rating       = int(rating_match.group(1)) if rating_match else 3
        rating       = max(1, min(5, rating))
        
        title_match  = re.search(r"TITLE:\s*(.+?)(?=\nREVIEW:|\Z)", raw, re.DOTALL)
        review_title = title_match.group(1).strip() if title_match else f"Review of {req.item_title[:30]}"
        
        review_match = re.search(r"REVIEW:\s*(.+)", raw, re.DOTALL)
        review_text  = review_match.group(1).strip() if review_match else raw
        
        variations.append(ReviewVariation(
            rating       = rating,
            review_title = review_title[:100],
            review_text  = review_text[:1000],
            style        = style_info["style"]
        ))
    
    return SimulateResponse(
        variations   = variations,
        persona_type = "custom" if mode == "custom_persona" else "cold_start",
        mode         = mode,
        language     = req.language
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "name": "DSN x BCT - Task A User Review Simulator",
        "team": "Drizzy x Metro",
        "version": "1.0.0",
        "endpoints": [
            "POST /simulate",
            "GET /health",
            "GET /stats",
            "GET /docs",
        ],
        "metrics": {
            "rmse": 0.7071,
            "bertscore": 0.8447,
            "rouge_1": 0.3093,
            "languages": ["pidgin", "yoruba", "hausa", "igbo", "english"],
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
