# 🎓 Hybrid E-Learning Recommender

A course recommendation engine combining three signals — **100% free, runs
entirely on your own machine, no paid API, no API key required.**

1. **Collaborative filtering** (matrix factorization via SVD) — "learners like you also took..."
2. **Semantic content-based filtering** (`sentence-transformers`, free local embeddings) — matches course *meaning*, not just keywords, to your history
3. **Goal-aware local re-ranking** — re-orders results based on a free-text goal like
   *"I want to get into backend dev but I'm scared of databases"*, and explains **why** each pick fits, using the same free local embeddings (no LLM API call at all)

### Standout features (for your report/demo)
- **Zero cost, zero API key** — everything runs locally. The only one-time download is a small (~80MB) open-source embedding model from Hugging Face on first run; after that it's fully offline.
- **Automatic fallback** — if there's no internet on first run (e.g. restricted campus wifi), it gracefully falls back to TF-IDF instead of crashing, and tells you which mode it's in.
- **Cold-start handling** — brand-new users with no history still get relevant recommendations, driven entirely by their stated goal.
- **MMR diversity re-ranking** — prevents the classic recommender failure mode of returning 8 near-duplicate courses from one category.
- **Explainable scoring** — every recommendation shows its CF score, content score, and goal score separately, which is great for a viva/interview where you'll be asked "how does it actually work?"

---

## 1. Project Structure

```
hybrid-elearning-recommender/
├── app.py                      # Streamlit app (this is the frontend)
├── data_generator.py           # Generates synthetic dataset (or swap for real data)
├── data/
│   ├── courses.csv
│   └── interactions.csv
├── recommender/
│   ├── collaborative.py        # SVD-based collaborative filtering
│   ├── content_based.py        # Free local semantic embeddings (+ TF-IDF fallback)
│   ├── hybrid.py                # Combines both + MMR diversity
│   └── local_reranker.py        # Free goal-aware re-ranking (no API)
└── requirements.txt
```

## 2. Local Setup

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Data is already generated, but to regenerate/customize it)
python data_generator.py

# 4. Run the app
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). On first
run, it downloads the free embedding model (~80MB, one time only) — after
that everything works fully offline.

## 3. Using Real Data Instead of Synthetic Data

Replace `data/courses.csv` and `data/interactions.csv`, keeping the same columns:

- `courses.csv`: `course_id, title, category, difficulty, skills, description, duration_hours, rating`
- `interactions.csv`: `user_id, course_id, completed, rating`

Good real sources: Kaggle's "Coursera Course Dataset", or scrape your own college's LMS course catalog (with permission) for a project that's uniquely yours — this scores extra points in a viva since nobody else will have the same dataset.

## 4. Deployment (pick one, all free tiers)

### Option A — Streamlit Community Cloud (easiest, free)
1. Push this project to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app" → select your repo → set main file to `app.py`.
3. Deploy. You'll get a public URL like `yourapp.streamlit.app` — put this straight on your resume.

### Option B — Hugging Face Spaces
1. Create a new Space → SDK: Streamlit.
2. Upload all project files (or connect your GitHub repo).
3. It auto-builds and deploys — no secrets/API keys needed.

### Option C — Render
1. New → Web Service → connect your GitHub repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

No environment variables or secrets are required for any of these — that's
one less thing that can go wrong during a live demo.

## 5. Resume Bullet Points (once deployed)

- *"Built a hybrid recommendation engine combining collaborative filtering (SVD) and semantic content-based filtering (sentence embeddings), with a goal-aware re-ranking layer that personalizes results to a user's stated career goal — fully self-hosted with zero API cost; deployed live at [link]."*
- *"Designed a diversity-aware re-ranking algorithm (MMR) to prevent category clustering in recommendations, improving perceived recommendation quality."*
- *"Implemented cold-start handling so new users receive relevant recommendations from goal-text alone, with zero interaction history."*

## 6. Ideas to Extend Further (bonus points)
- Add an A/B evaluation: compare hybrid vs. CF-only vs. CB-only using precision@k on a held-out test split.
- Add a feedback loop: let users thumbs-up/down recommendations and retrain.
- If you want a true generative re-ranker without paying for an API, run [Ollama](https://ollama.com) locally with a small open model (e.g. `llama3.2:1b`) — it's free, local, and you can point `local_reranker.py` at it via Ollama's local HTTP API instead of Anthropic's.
