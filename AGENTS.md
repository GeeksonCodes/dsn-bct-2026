# DSN x BCT LLM Agent Challenge — Full Engineering Brief
> Read this entire document before writing any code.
> This is the complete context for building the FastAPI backends, Web UI, and Docker setup.

---

## 1. Project Overview

**Competition:** DSN x BCT LLM Agent Challenge — Hackathon 3.0
**Deadline:** May 24, 2026 — end of day
**Goal:** Two LLM-based agents for user behaviour modelling and recommendation
**Team:** Timilehin (engineering) + Teammate (solution paper)
**Repo:** GitHub org `dsn-bct-2026`, repo `dsn-bct-llm-agent`

---

## 2. The Two Tasks

### Task A — User Modeling
Simulate what a specific user would write as a review for an unseen product.

**Input:** user_id (optional) + item details
**Output:** star rating (1-5) + review title + review text

**Key behaviour:**
- If user_id provided AND exists in persona_cache → use cached persona
- If user_id provided but NOT in cache → run Agent 1 (Gemini Flash) to build persona, then Agent 2 to generate
- If no user_id → cold-start mode, generate generic but persona-conditioned review

### Task B — Recommendation
Return personalised product recommendations for a user.

**Input:** user answers to 3 questions OR user_id
**Output:** ranked list of 5 products with Nigerian-contextualised explanations

**Key behaviour:**
- Cold-start (no user_id, 92.3% of users): 3-question elicitation → persona query → FAISS retrieval → blend scoring → LLM reranker
- 1-shot (user has 1 review): extract persona from single review via LLM → retrieve → rerank
- History-based (user has ≥2 reviews): build persona from history → retrieve → rerank

---

## 3. Repository Structure

```
dsn-bct-llm-agent/
├── app/
│   ├── task_a/
│   │   ├── main.py              ← Task A FastAPI (BUILD THIS)
│   │   ├── Dockerfile           ← Task A Docker (BUILD THIS)
│   │   ├── requirements.txt     ← Task A deps (BUILD THIS)
│   │   ├── rag_index.faiss      ← FAISS index (EXISTS, DO NOT MODIFY)
│   │   ├── rag_meta.pkl         ← RAG metadata (EXISTS, DO NOT MODIFY)
│   │   ├── rich_users.csv       ← cleaned user data (EXISTS)
│   │   ├── persona_cache/       ← cached persona JSONs (EXISTS)
│   │   └── item_cache/          ← cached item descriptions (EXISTS)
│   └── task_b/
│       ├── main.py              ← Task B FastAPI (BUILD THIS)
│       ├── Dockerfile           ← Task B Docker (BUILD THIS)
│       ├── requirements.txt     ← Task B deps (BUILD THIS)
│       ├── items.index          ← FAISS index 112k items (EXISTS)
│       └── item_meta.pkl        ← item metadata (EXISTS)
├── frontend/
│   └── index.html               ← Shared Web UI (BUILD THIS)
├── docker-compose.yml            ← Connects everything (BUILD THIS)
├── paper/                        ← LaTeX paper (teammate's job)
└── README.md                     ← Setup instructions (BUILD THIS)
```

---

## 4. Task A — Technical Details

### Pipeline (from notebook: task_a_user_modeling_pipeline.ipynb)

```
User ID + Item Details
        │
        ▼
PersonaCache.load(user_id)
        │
   Found?
   YES → load PersonaDossier from JSON file
   NO  → run_agent1_analyst(user_id, texts) → PersonaDossier
        │
        ▼
ItemCardConstructor.build(asin, title, description)
        │
        ▼
retrieve_for_agent2(
    query = item_title + description,
    user_id = user_id,           ← filters to this user's reviews
    rag_index = rag_index,
    rag_metadata = rag_metadata
)
        │
        ▼
run_agent2_with_rag(
    dossier = PersonaDossier,
    item_asin, item_title, item_description,
    rag_examples = retrieved reviews,
    nigerian_mode = True/False
)
        │
        ▼
SimulatedReview { rating, title, review_text }
```

### Key Functions (already built in notebook)
```python
simulate_review_two_agent(
    user_id,          # str or None
    texts,            # pd.DataFrame — rich_df filtered to user
    item_asin,        # str
    item_title,       # str  
    item_description, # str
    nigerian_mode,    # bool
    flash_model,      # "gemini-2.0-flash"
    pro_model,        # "gemini-2.0-flash"
    persona_cache,    # PersonaCache instance
    rag_index,        # faiss.Index
    rag_metadata,     # list of dicts
    embedder          # SentenceTransformer
)
```

### Artifacts Task A Loads At Startup
```python
# Load once at startup, reuse for every request
rich_df      = pd.read_csv("rich_users.csv")
rag_index    = faiss.read_index("rag_index.faiss")
with open("rag_meta.pkl", "rb") as f:
    rag_metadata = pickle.load(f)
persona_cache = PersonaCache(cache_dir="persona_cache/")
embedder      = SentenceTransformer("all-MiniLM-L6-v2")
```

### Task A API Endpoint
```
POST /simulate
Content-Type: application/json

Request:
{
    "user_id": "AGKHLEW2SOWHNMFQIJGBECAF7INQ",  // optional
    "item_asin": "B00YQ6X8EO",
    "item_title": "Lavender Body Lotion",
    "item_description": "A lightweight daily moisturiser with calming lavender scent.",
    "nigerian_mode": true
}

Response:
{
    "rating": 4,
    "review_title": "Good but packaging could be better",
    "review_text": "I go try am and e good o! The scent no too strong...",
    "persona_type": "Balanced",
    "mode": "history_based",
    "user_id": "AGKHLEW2SOWHNMFQIJGBECAF7INQ"
}
```

---

## 5. Task B — Technical Details

### Pipeline (from notebook: task_b_recommendation_pipeline.ipynb)

```
User Input
    │
    ├── Has user_id with ≥2 reviews?
    │       → recommend_from_history()
    │
    ├── Has user_id with 1 review?
    │       → extract_oneshot_persona() → LLM extracts persona
    │         → retrieve_items() → post_filter() → llm_rerank_safe()
    │
    └── No user_id (cold-start, 92.3% of users)
            → 3 answers from request body
            → build_retrieval_query() → positive framing
            → retrieve_items() → post_filter() → llm_rerank_safe()
```

### Blend Scoring (CONFIRMED OPTIMAL from ablation)
```python
final_score = 0.5 * semantic_similarity + 0.5 * popularity_score
# This was confirmed optimal across 9 experiments
# NDCG@10: 0.4210 | Hit@10: 0.6600
```

### Key Functions (already built in notebook)
```python
def retrieve_items(query_text, n_results=20, min_reviews=3):
    # Embeds query with MiniLM
    # Searches FAISS index (items.index)
    # Returns blended semantic + popularity scored items

def post_filter(items, avoid_keyword):
    # Removes items mentioning avoided ingredient in title

def build_retrieval_query(answers):
    # Converts elicitation answers to positive-framed query
    # IMPORTANT: never include negations in query
    # "avoid alcohol" → "gentle natural fragrance-free"
    # Returns: (query_string, avoid_keyword)

def llm_rerank_safe(persona_query, candidates, nigerian_mode=True):
    # Shuffles candidates (prevents positional bias)
    # Calls Gemini 2.0 Flash with Nigerian context
    # Has 3-attempt retry loop
    # Has word-overlap hallucination fix
    # Falls back to rule-based if API fails

def recommend_multilingual(language, nigerian_mode):
    # Handles: pidgin, yoruba, hausa, igbo, english
    # Runs elicitation + retrieval + reranker
    # Returns language-appropriate explanations
```

### Artifacts Task B Loads At Startup
```python
# Load once at startup
index     = faiss.read_index("items.index")     # 112k items, dim=384
item_meta = pickle.load(open("item_meta.pkl", "rb"))
embedder  = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Build pop_lookup from item_meta
max_reviews = item_meta["review_count"].quantile(0.95)
item_meta["pop_score"] = (
    item_meta["review_count"].clip(upper=max_reviews) / max_reviews
) * item_meta["avg_rating"] / 5.0
pop_lookup = dict(zip(item_meta["parent_asin"], item_meta["pop_score"]))
```

### Task B API Endpoints
```
POST /recommend
Content-Type: application/json

Request (cold-start):
{
    "user_id": null,
    "product_type": "skincare",
    "priority": "natural ingredients",
    "avoid": "alcohol",
    "language": "pidgin",
    "nigerian_mode": true
}

Request (returning user):
{
    "user_id": "AGKHLEW2SOWHNMFQIJGBECAF7INQ",
    "product_type": null,
    "priority": null,
    "avoid": null,
    "language": "pidgin",
    "nigerian_mode": true
}

Response:
{
    "reasoning": "This user dey look for...",
    "recommendations": [
        {
            "rank": 1,
            "asin": "B00YQ6X8EO",
            "title": "Gentle Lavender Moisturiser",
            "avg_rating": 4.5,
            "review_count": 120,
            "explanation": "E good o! Natural ingredients, price reasonable..."
        }
    ],
    "persona_type": "cold_start",
    "language": "pidgin"
}

GET /health
Response: {"status": "ok", "ndcg_at_10": 0.4210, "hit_rate_10": 0.6600}
```

---

## 6. LLM Configuration

### Gemini Setup (use this exact pattern)
```python
from google import genai  # NOT google.generativeai (deprecated)
import time

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def generate(prompt, max_retries=5):
    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash"]
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                if "429" in str(e):
                    wait = 15 * (attempt + 1)
                    time.sleep(wait)
                else:
                    break
    return None
```

### Embedder Setup
```python
from sentence_transformers import SentenceTransformer
# Use all-MiniLM-L6-v2 — DO NOT change model
# BGE-large was tested but showed no NDCG improvement
# MiniLM is faster and equivalent quality for this dataset
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
# Use CPU — CUDA kernel mismatch issues on some environments
```

---

## 7. Nigerian Contextualisation

This is a core feature — judges award bonus marks for it.

### Language Prompts
```python
LANGUAGE_PROMPTS = {
    "pidgin": {
        "name": "Nigerian Pidgin",
        "greeting": "Wetin you need today?",
        "system": """
Respond naturally in Nigerian Pidgin mixed with English.
Sound like a trusted friend, not a salesperson.
Example: "This one e good o! E no go break your pocket.
Many people don use am, dem all happy. I go advise make you try am!"
"""
    },
    "yoruba": {
        "name": "Yoruba-English",
        "greeting": "Ẹ káàbọ̀! How can I help you?",
        "system": """
Mix Yoruba phrases naturally with English.
Example: "Ẹ káàbọ̀! This product o dára gan-an.
Ó tọ́ iye rẹ̀ — worth every naira. Ẹ gbìyànjú!"
"""
    },
    "hausa": {
        "name": "Hausa-English",
        "greeting": "Sannu! How can I help you?",
        "system": """
Mix Hausa phrases naturally with English.
Example: "Sannu! Wannan kaya yana da kyau sosai.
Mai rahusa ne — affordable and effective. Ina ba da shawara!"
"""
    },
    "igbo": {
        "name": "Igbo-English",
        "greeting": "Nnọọ! How can I help you?",
        "system": """
Mix Igbo phrases naturally with English.
Example: "Nnọọ! Ihe a dị mma nke ọma.
O dị ire ire — good value. Nwanne, I ga-amasị ya!"
"""
    },
    "english": {
        "name": "Nigerian English",
        "greeting": "Hello! How can I help you today?",
        "system": """
Use Nigerian English — practical, value-conscious, communal.
Reference everyday Nigerian consumer concerns naturally.
"""
    }
}
```

### Rules
- NEVER force Pidgin on every sentence
- Natural code-switching only
- Reference Nigerian brands/stores when relevant
- Always emphasise value for money

---

## 8. Known Problems To Handle

### Problem 1 — No User History (Task A cold-start)
**Issue:** Without reviewer history, Agent 2 generates generic reviews.
**Solution:** Even without user_id, still run Agent 2 with a synthetic persona:
```python
# If no user_id provided, create a generic Nigerian beauty consumer persona
synthetic_persona = PersonaDossier(
    avg_rating=4.0,
    persona_type="Balanced",
    writing_style="conversational",
    nigerian_signals=NigerianSignals(
        price_conscious=True,
        community_oriented=True
    )
)
```

### Problem 2 — Positional Bias In LLM Reranker
**Issue:** LLM favours items at top of candidate list.
**Solution:** Always shuffle candidates before passing to LLM:
```python
import random
random.shuffle(candidates)  # ALWAYS do this before reranker
```

### Problem 3 — LLM Hallucinating Item Titles
**Issue:** Reranker invents product names not in candidate list.
**Solution:** Word overlap matching maps hallucinated title back to real item:
```python
def match_to_candidate(llm_title, candidates):
    llm_words = set(llm_title.lower().split())
    best_match = max(candidates, key=lambda c:
        len(llm_words & set(c["title"].lower().split()))
    )
    return best_match
```

### Problem 4 — API Rate Limits (429 errors)
**Issue:** Gemini free tier exhausts quickly.
**Solution:**
- Use retry loop with exponential backoff
- Fall back to `gemini-1.5-flash` if `gemini-2.0-flash` quota exhausted
- Return rule-based fallback if all retries fail (never crash)

### Problem 5 — Negations In Retrieval Query
**Issue:** "avoid alcohol" in query matches items that CONTAIN alcohol.
**Solution:** Never put negations in the embedding query. Reframe positively:
```python
# WRONG:
query = "skincare that avoids alcohol"

# CORRECT:
query = "gentle natural skincare fragrance-free"
avoid = "alcohol"  # used for post-retrieval filtering only
```

### Problem 6 — FAISS Index Dimension Mismatch
**Issue:** If model changes, new embeddings (1024d) won't match old index (384d).
**Solution:** Always verify at startup:
```python
assert index.d == embedder.get_sentence_embedding_dimension(), \
    f"Dimension mismatch: index={index.d}, model={embedder.get_sentence_embedding_dimension()}"
```

---

## 9. Docker Configuration

### Task A Container
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY rag_index.faiss .
COPY rag_meta.pkl .
COPY rich_users.csv .
COPY persona_cache/ ./persona_cache/
COPY item_cache/ ./item_cache/
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Task B Container
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY items.index .
COPY item_meta.pkl .
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
```

### Docker Compose
```yaml
version: "3.8"
services:
  task_a:
    build: ./app/task_a
    ports:
      - "8001:8001"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    restart: unless-stopped

  task_b:
    build: ./app/task_b
    ports:
      - "8002:8002"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    restart: unless-stopped

  frontend:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./frontend:/usr/share/nginx/html
    depends_on:
      - task_a
      - task_b
    restart: unless-stopped
```

---

## 10. Web UI Requirements

### Single Page — Two Tabs

**Tab 1: Product Review Simulator (Task A)**
- Input fields: Item Name, Item Description, User ID (optional)
- Language toggle: English / Nigerian Pidgin
- Submit button → calls `POST http://localhost:8001/simulate`
- Output: Star rating display + generated review text

**Tab 2: Product Recommender (Task B)**
- Language selector: Pidgin / Yoruba / Hausa / Igbo / English
- User ID field (optional)
- If no user ID: show 3 elicitation questions
  - Q1: What products do you use most?
  - Q2: What matters most when buying?
  - Q3: Any ingredients you avoid?
- Submit button → calls `POST http://localhost:8002/recommend`
- Output: 5 recommendation cards with rating, review count, explanation

### Design Requirements
- Mobile-friendly
- DSN/BCT branding colours (blue and gold)
- Nigerian flag accent
- Loading spinner during API calls
- Error handling — show friendly message if API fails
- Both tabs visible without scrolling on desktop

---

## 11. Evaluation Results (For README and Demo)

### Task A Final Results
```
RMSE        : 0.7071  (strong threshold < 0.80 ✅)
BERTScore   : 0.8447
ROUGE-1     : 0.3093
Trait cons. : 0.4750
Users eval  : 20
Architecture: Two-agent (Flash persona + Flash+RAG generation)
```

### Nigerian Mode Comparison (Task A)
```
Metric        Standard   Nigerian   
RMSE          0.7071     0.7416    
ROUGE-1       0.3093     0.3231 ✅ improved
ROUGE-L       0.1581     0.1647 ✅ improved
BERTScore     0.8447     0.8386
Trait cons.   0.4750     0.4875 ✅ improved
```

### Task B Final Results
```
NDCG@10     : 0.4210  (100 users)
Hit Rate@10 : 0.6600  (win zone ≥ 0.65 ✅)
Optimal blend: 50% semantic + 50% popularity
Languages   : Pidgin, Yoruba, Hausa, Igbo, English
User types  : cold-start, 1-shot, history-based
```

### Task B Ablation Study
```
Strategy                    NDCG@10   Hit@10
Pure semantic               0.0808    0.1600
Sem:0.7 + Pop:0.3           0.3110    0.5400
Sem:0.5 + Pop:0.5 (BEST)    0.4210    0.6600
Sem:0.4 + Pop:0.6           0.3134    0.6200
Blend only vs + reranker:
  Blend only                0.3448    0.6667
  + LLM reranker            0.3300    0.6667
  (reranker optimises human eval, not ranking NDCG)
```

---

## 12. Environment Variables

```bash
# Required for both containers
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional — HuggingFace token for BERTScore
HF_TOKEN=your_huggingface_token_here
```

Create a `.env` file in the repo root:
```
GOOGLE_API_KEY=your_key_here
```

Add to `.gitignore`:
```
.env
app/task_a/rag_index.faiss
app/task_a/rag_meta.pkl
app/task_a/rich_users.csv
app/task_a/persona_cache/
app/task_a/item_cache/
app/task_b/items.index
app/task_b/item_meta.pkl
```

---

## 13. Build Order

```
Step 1: app/task_b/requirements.txt
Step 2: app/task_b/main.py
Step 3: app/task_b/Dockerfile
Step 4: app/task_a/requirements.txt
Step 5: app/task_a/main.py
Step 6: app/task_a/Dockerfile
Step 7: frontend/index.html
Step 8: docker-compose.yml
Step 9: README.md
Step 10: Test — docker-compose up --build
Step 11: Verify both APIs respond at :8001 and :8002
Step 12: Verify frontend loads at :80
```

---

## 14. Paper Context (For Teammate)

Papers cited:
- Wang et al. (2023) — RecAgents
- Zhang et al. (2024) — Agents4Rec (SIGIR '24)
- Liu et al. (2023) — Is ChatGPT a Good Recommender? (CIKM '23)

Key findings to cite:
- Few-shot behaviour injection reduces RMSE (Liu et al.)
- Positional bias in LLM reranking — must shuffle (Liu et al.)
- ROUGE artificially low for LLM-generated text — BERTScore preferred (Liu et al.)
- Three-tier memory improves fidelity (Wang et al.) — mentioned as future work
- Popularity bias correction via algorithmic randomness (Wang et al.)
- Cold-start via explicit profiling (Zhang et al.)

Related Work section already drafted — see conversation history with Claude.

---

## 15. Submission Checklist

```
□ Task A containerised app running at POST /simulate
□ Task B containerised app running at POST /recommend
□ Web UI showing both tasks
□ docker-compose up --build works from clean clone
□ README with setup instructions
□ GitHub repo clean — no large files committed
□ Solution paper 4-8 pages (teammate)
□ Submit via form before May 24 midnight
```

---

## 16. Multi-Provider API Key Pool

Both Task A and Task B must implement a multi-provider key pool
that automatically rotates across providers when rate limits hit.

### Providers In Priority Order
1. Gemini 2.0 Flash (primary — best quality)
2. Gemini 1.5 Flash (fallback — separate quota)
3. Cerebras DeepSeek (fallback — ultra-fast distill-llama-70b)
4. Groq llama-3.3-70b-versatile (fallback — very fast, free tier)

### Environment Variables Required
```
GOOGLE_API_KEY_1=your_primary_gemini_key
GOOGLE_API_KEY_2=your_secondary_gemini_key
CEREBRAS_API_KEY=your_cerebras_key
GROQ_API_KEY=your_groq_key
```


### Key Manager Implementation
```python
import time
import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

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
        # Gemini primary
        key1 = os.environ.get("GOOGLE_API_KEY_1")
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
            raise ValueError("No API keys configured")

        print(f"Key pool initialised: {len(self.keys)} providers")
        for k in self.keys:
            print(f"  ✅ {k.provider.value}")


    def generate(self, prompt: str) -> Optional[str]:
        """
        Try each provider in order.
        Track latency and errors per key.
        """
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

                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    key_stat.errors_429 += 1
                    print(f"⚠️  {key_stat.provider.value} rate limited — trying next")
                    # Mark as exhausted for daily quota errors
                    if "quota" in str(e).lower():
                        key_stat.is_exhausted = True
                    else:
                        time.sleep(15)
                else:
                    key_stat.errors_other += 1
                    print(f"❌ {key_stat.provider.value} error: {str(e)[:100]}")

        # All providers failed — return None (fallback handled upstream)
        print("❌ All providers exhausted")
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


# Initialise once at startup
key_manager = MultiProviderKeyManager()

# Use everywhere instead of direct generate() calls
def generate(prompt: str) -> Optional[str]:
    return key_manager.generate(prompt)
```

### Additional Endpoint — Key Performance Stats
Both Task A and Task B must expose this endpoint:

```
GET /stats
Response:
{
    "providers": [
        {
            "provider": "gemini-2.0-flash",
            "requests_made": 47,
            "errors_429": 2,
            "avg_latency_s": 1.823,
            "error_rate_pct": 4.3,
            "is_exhausted": false
        },
        {
            "provider": "gemini-1.5-flash", 
            "requests_made": 3,
            "errors_429": 0,
            "avg_latency_s": 1.654,
            "error_rate_pct": 0.0,
            "is_exhausted": false
        },
        {
            "provider": "groq-llama-3.3-70b",
            "requests_made": 0,
            "errors_429": 0,
            "avg_latency_s": 0.0,
            "error_rate_pct": 0.0,
            "is_exhausted": false
        }
    ],
    "total_requests": 50,
    "active_provider": "gemini-2.0-flash"
}
```

### Environment Variables (update .env)
```
GOOGLE_API_KEY_1=your_primary_gemini_key
GOOGLE_API_KEY_2=your_secondary_gemini_key
CEREBRAS_API_KEY=your_cerebras_key
GROQ_API_KEY=your_groq_key
```


### Get Groq API Key (Free)
1. Go to console.groq.com
2. Sign up free
3. API Keys → Create API Key
4. Copy key starting with gsk_...
```

---

*Last updated: May 20, 2026 — Day 10 of 14*
*Timilehin handles: FastAPI, Docker, Web UI*
*Teammate handles: Solution paper*
*Repo: dsn-bct-2026/dsn-bct-llm-agent*
