# 🇳🇬 DSN x BCT LLM Agent Challenge — Production Suite

An advanced, containerised dual-agent system for user behavior modeling, synthetic review simulation, and personalised product recommendation. Features a multi-provider key manager for automatic failover rotation and real-time telemetry tracking.

---

## 📊 Evaluation & Empirical Results

### Task A — User Review Simulator
Simulates highly faithful product reviews conditioned on a user's historical profile and retrieved context.

*   **Architecture**: Two-Stage Chain (Agent 1: Flash Persona Discovrer + Agent 2: Flash RAG-conditioned Review generator).
*   **Evaluation Performance Metrics**:
    *   **RMSE**: `0.7071` (Strong performance threshold `< 0.80` ✅)
    *   **BERTScore**: `0.8447`
    *   **ROUGE-1**: `0.3093`
    *   **Trait Consistency**: `0.4750`

#### 🇳🇬 Standard vs. Nigerian Mode (Task A)
Integrating cultural nuances, local code-switching (Pidgin/Yoruba/Igbo/Hausa), and value-conscious consumer traits significantly improved text fidelity:

| Metric | Standard Mode | Nigerian Mode | Status |
| :--- | :--- | :--- | :--- |
| **RMSE** | `0.7071` | `0.7416` | Robust variance stability |
| **ROUGE-1** | `0.3093` | **`0.3231`** | **Improved** ✅ |
| **ROUGE-L** | `0.1581` | **`0.1647`** | **Improved** ✅ |
| **BERTScore** | `0.8447` | `0.8386` | High semantic alignment |
| **Trait Consistency** | `0.4750` | **`0.4875`** | **Improved** ✅ |

---

### Task B — Personalised Recommender
Generates top-5 product recommendations utilizing hybrid algorithmic scoring and multilingual Nigerian contextualisation.

*   **Optimal Retrieval Strategy**: Blended Scoring (50% Semantic Cosine Similarity + 50% Algorithmic Popularity).
*   **Optimal Performance Metrics**:
    *   **NDCG@10**: `0.4210`
    *   **Hit Rate@10**: `0.6600` (Exceeds competition target `≥ 0.65` ✅)

#### 🧪 Task B Ablation Studies (Ablation Experiments)

| Strategy | NDCG@10 | Hit@10 | Status |
| :--- | :--- | :--- | :--- |
| **Pure Semantic Search** | `0.0808` | `0.1600` | High niche bias |
| **Semantic: 0.7 + Popularity: 0.3** | `0.3110` | `0.5400` | Baseline |
| **Semantic: 0.5 + Popularity: 0.5** | **`0.4210`** | **`0.6600`** | **Optimal Configuration** (BEST) 🏆 |
| **Semantic: 0.4 + Popularity: 0.6** | `0.3134` | `0.6200` | High popularity bias |
| **Blend Only** (No Rerank) | `0.3448` | `0.6667` | High raw recall |
| **Blend + LLM Reranker** | `0.3300` | `0.6667` | **Optimises User Reasoning & Explanations** 🌟 |

---

## 🔑 Multi-Provider API Key Pool (Section 16)

To guarantee production-grade availability and bypass API rate limits (`429 Too Many Requests`), the system implements a **live active-passive key manager** rotating across:
1.  **Gemini 2.0 Flash** (Primary — best quality)
2.  **Gemini 1.5 Flash** (Fallback — separate rate quota)
3.  **Groq Llama-3.3-70b-versatile** (Fallback — low latency, free tier)

---

## 🛠️ Quick Start

### 1. Configure Keys (`.env`)
Create a `.env` file at the repository root:
```env
GOOGLE_API_KEY_1=your_primary_gemini_key
GOOGLE_API_KEY_2=your_secondary_gemini_key  
GROQ_API_KEY=your_groq_key
```

### 2. Run via Docker Compose (Recommended)
Launch the entire system (Task A, Task B, and Nginx Frontend) in one command:
```bash
docker-compose up --build
```
*   **Web UI**: Open `http://localhost/` in your browser.
*   **Task A Backend**: Available at `http://localhost:8001/`
*   **Task B Backend**: Available at `http://localhost:8002/`

### 3. Run Locally (Alternative)
Set up a python virtual environment and run tasks:
```bash
# 1. Create and activate venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r app/task_a/requirements.txt -r app/task_b/requirements.txt

# 3. Start Task A
cd app/task_a
$env:GOOGLE_API_KEY_1="your_key"  # Windows PowerShell
uvicorn main:app --host 0.0.0.0 --port 8001

# 4. Start Task B (in another terminal)
cd app/task_b
$env:GOOGLE_API_KEY_1="your_key"  # Windows PowerShell
uvicorn main:app --host 0.0.0.0 --port 8002
```

---

## 📡 API Reference

### Task A: POST `/simulate`
Simulate a product review based on a user persona.
```json
// POST http://localhost:8001/simulate
{
  "user_id": "AHBWH2LBU3NFLD46GKJKIBAHKXEQ",
  "item_asin": "B00YQ6X8EO",
  "item_title": "Lavender Body Lotion",
  "item_description": "Moisturiser with lavender scent.",
  "nigerian_mode": true
}
```

### Task B: POST `/recommend`
Retrieve personalised product recommendations.
```json
// POST http://localhost:8002/recommend
{
  "user_id": null,
  "product_type": "skincare",
  "priority": "natural ingredients",
  "avoid": "alcohol",
  "language": "pidgin",
  "nigerian_mode": true
}
```

### Shared Telemetry Endpoint: GET `/stats`
Retrieves key manager pool health and provider loads.
```json
// GET http://localhost:8001/stats  (Also at :8002)
{
  "providers": [
    {
      "provider": "gemini-2.0-flash",
      "requests_made": 47,
      "errors_429": 2,
      "avg_latency_s": 1.823,
      "error_rate_pct": 4.3,
      "is_exhausted": false
    }
  ],
  "total_requests": 47,
  "active_provider": "gemini-2.0-flash"
}
```
