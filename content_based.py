"""
Content-based filtering using SEMANTIC embeddings — free and fully local,
no API key, no cost.

Primary path: `sentence-transformers` (all-MiniLM-L6-v2, ~80MB, downloads
once from Hugging Face then runs offline forever). This understands meaning,
not just keyword overlap — e.g. it knows "REST APIs" and "backend services"
are related even with zero shared words, which plain TF-IDF cannot do.

Fallback path: if sentence-transformers isn't installed, the model can't be
downloaded, or the DISABLE_SEMANTIC env var is set — this automatically
falls back to TF-IDF so the app never crashes. TF-IDF also uses far less
RAM, which matters on free-tier hosting (Render free = 512MB) where the
PyTorch dependency behind sentence-transformers can be too heavy.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

_FORCE_TFIDF = os.environ.get("DISABLE_SEMANTIC", "").lower() in ("1", "true", "yes")

if not _FORCE_TFIDF:
    try:
        from sentence_transformers import SentenceTransformer
        _ST_AVAILABLE = True
    except ImportError:
        _ST_AVAILABLE = False
else:
    _ST_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer


class ContentBasedRecommender:
    def __init__(self, use_semantic=True):
        self.use_semantic = use_semantic and _ST_AVAILABLE
        self.model = None
        self.vectorizer = None
        self.course_vectors = None
        self.courses_df = None
        self.course_id_to_idx = {}

        if self.use_semantic:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Could not load sentence-transformers model ({e}); falling back to TF-IDF.")
                self.use_semantic = False

        if not self.use_semantic:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)

    def fit(self, courses_df: pd.DataFrame):
        self.courses_df = courses_df.reset_index(drop=True)
        corpus = (
            self.courses_df["title"] + " "
            + self.courses_df["description"] + " "
            + self.courses_df["skills"]
        ).tolist()

        if self.use_semantic:
            self.course_vectors = self.model.encode(corpus, show_progress_bar=False)
        else:
            self.course_vectors = self.vectorizer.fit_transform(corpus)

        self.course_id_to_idx = {
            cid: i for i, cid in enumerate(self.courses_df["course_id"])
        }
        return self

    def _embed_query(self, text: str):
        if self.use_semantic:
            return self.model.encode([text], show_progress_bar=False)
        return self.vectorizer.transform([text])

    def score_all_courses(self, taken_course_ids):
        if not taken_course_ids:
            return {cid: 0.0 for cid in self.course_id_to_idx}

        idxs = [self.course_id_to_idx[c] for c in taken_course_ids if c in self.course_id_to_idx]
        if not idxs:
            return {cid: 0.0 for cid in self.course_id_to_idx}

        if self.use_semantic:
            taste_vector = np.mean(self.course_vectors[idxs], axis=0, keepdims=True)
        else:
            taste_vector = np.asarray(self.course_vectors[idxs].mean(axis=0))

        sims = cosine_similarity(taste_vector, self.course_vectors).flatten()
        return {cid: float(sims[idx]) for cid, idx in self.course_id_to_idx.items()}

    def score_against_goal_text(self, goal_text: str):
        goal_vector = self._embed_query(goal_text)
        sims = cosine_similarity(goal_vector, self.course_vectors).flatten()
        return {cid: float(sims[idx]) for cid, idx in self.course_id_to_idx.items()}

    def top_matching_skills(self, goal_text: str, course_skills: str, top_k=3):
        """
        Used by the local re-ranker to build human-readable explanations
        without needing a generative LLM: finds which of the course's listed
        skills are most semantically related to the goal text.
        """
        skills = [s.strip() for s in course_skills.split(",") if s.strip()]
        if not skills:
            return []

        if self.use_semantic:
            goal_vec = self.model.encode([goal_text], show_progress_bar=False)
            skill_vecs = self.model.encode(skills, show_progress_bar=False)
        else:
            combined = self.vectorizer.transform([goal_text] + skills)
            goal_vec, skill_vecs = combined[0:1], combined[1:]

        sims = cosine_similarity(goal_vec, skill_vecs).flatten()
        ranked = sorted(zip(skills, sims), key=lambda x: x[1], reverse=True)
        return [s for s, score in ranked[:top_k] if score > 0.15]
