"""
Collaborative Filtering via matrix factorization (truncated SVD).

Why SVD instead of `surprise`/`implicit`: those libraries need compiled
C extensions that are painful to deploy on free hosting (Streamlit Cloud,
HF Spaces). scipy's svds gives the same core idea (latent user/item
factors) with zero deployment headaches.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class CollaborativeRecommender:
    def __init__(self, n_factors=20):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_id_map = {}
        self.course_id_map = {}
        self.reverse_course_map = {}
        self.global_mean = 0.0

    def fit(self, interactions_df: pd.DataFrame):
        # Build an implicit "affinity" score: completed + rating matters more
        # than just viewing. Missing rating (not completed) still counts weakly.
        df = interactions_df.copy()
        df["score"] = df.apply(
            lambda r: (r["rating"] if pd.notna(r["rating"]) else 3.0)
            + (1.0 if r["completed"] else 0.0),
            axis=1,
        )

        user_ids = sorted(df["user_id"].unique())
        course_ids = sorted(df["course_id"].unique())
        self.user_id_map = {uid: i for i, uid in enumerate(user_ids)}
        self.course_id_map = {cid: i for i, cid in enumerate(course_ids)}
        self.reverse_course_map = {i: cid for cid, i in self.course_id_map.items()}

        rows = df["user_id"].map(self.user_id_map)
        cols = df["course_id"].map(self.course_id_map)
        vals = df["score"].values

        matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(user_ids), len(course_ids)),
        )

        self.global_mean = vals.mean()

        k = min(self.n_factors, min(matrix.shape) - 1)
        k = max(k, 2)
        U, sigma, Vt = svds(matrix.astype(float), k=k)
        sigma = np.diag(sigma)

        self.user_factors = U.dot(sigma)
        self.item_factors = Vt.T
        self._interactions_df = df
        return self

    def score_all_courses(self, user_id):
        """Returns {course_id: predicted_score} for every course, for one user."""
        if user_id not in self.user_id_map:
            # Cold-start user: no history yet -> flat scores (content-based will dominate)
            return {cid: self.global_mean for cid in self.course_id_map}

        u_idx = self.user_id_map[user_id]
        preds = self.user_factors[u_idx].dot(self.item_factors.T)
        return {
            self.reverse_course_map[i]: float(preds[i])
            for i in range(len(preds))
        }

    def already_taken(self, user_id):
        if not hasattr(self, "_interactions_df"):
            return set()
        return set(
            self._interactions_df.loc[
                self._interactions_df["user_id"] == user_id, "course_id"
            ]
        )
