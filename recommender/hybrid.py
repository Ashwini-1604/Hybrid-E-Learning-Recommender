"""
Combines collaborative filtering, content-based filtering, and goal-intent
scores into one ranked list, then applies a lightweight diversity re-ranker
(Maximal Marginal Relevance) so recommendations aren't all near-duplicates
of each other — a common failure mode in student recommender projects.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def _normalize(scores_dict):
    if not scores_dict:
        return {}
    values = np.array(list(scores_dict.values())).reshape(-1, 1)
    if values.max() == values.min():
        return {k: 0.5 for k in scores_dict}
    scaled = MinMaxScaler().fit_transform(values).flatten()
    return dict(zip(scores_dict.keys(), scaled))


class HybridRecommender:
    def __init__(self, cf_model, cb_model, w_cf=0.4, w_cb=0.3, w_goal=0.3):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.w_cf = w_cf
        self.w_cb = w_cb
        self.w_goal = w_goal

    def recommend(self, user_id, courses_df, goal_text="", top_n=8, diversity=0.3):
        taken = self.cf_model.already_taken(user_id)

        cf_scores = _normalize(self.cf_model.score_all_courses(user_id))
        cb_scores = _normalize(self.cb_model.score_all_courses(taken))

        if goal_text.strip():
            goal_scores = _normalize(self.cb_model.score_against_goal_text(goal_text))
        else:
            goal_scores = {cid: 0.0 for cid in cf_scores}

        combined = {}
        for cid in cf_scores:
            if cid in taken:
                continue  # never recommend something they already took
            combined[cid] = (
                self.w_cf * cf_scores.get(cid, 0)
                + self.w_cb * cb_scores.get(cid, 0)
                + self.w_goal * goal_scores.get(cid, 0)
            )

        # Rank candidates, keep a generous shortlist for diversity re-ranking / LLM re-ranking
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        shortlist = ranked[: max(top_n * 3, 15)]

        diversified = self._mmr_rerank(shortlist, courses_df, top_n, diversity)

        result_rows = []
        for cid, score in diversified:
            course = courses_df.loc[courses_df["course_id"] == cid].iloc[0]
            result_rows.append({
                "course_id": cid,
                "title": course["title"],
                "category": course["category"],
                "difficulty": course["difficulty"],
                "description": course["description"],
                "skills": course["skills"],
                "duration_hours": course["duration_hours"],
                "rating": course["rating"],
                "hybrid_score": round(score, 4),
                "cf_score": round(cf_scores.get(cid, 0), 3),
                "cb_score": round(cb_scores.get(cid, 0), 3),
                "goal_score": round(goal_scores.get(cid, 0), 3),
            })
        return pd.DataFrame(result_rows)

    def _mmr_rerank(self, shortlist, courses_df, top_n, diversity):
        """
        Maximal Marginal Relevance: greedily picks the next best item that
        is both high-scoring AND dissimilar (different category) from what's
        already been picked, so the final list isn't 8 near-identical courses.
        """
        selected = []
        remaining = shortlist.copy()
        cat_lookup = dict(zip(courses_df["course_id"], courses_df["category"]))

        while remaining and len(selected) < top_n:
            best_idx, best_val = 0, -1e9
            for i, (cid, score) in enumerate(remaining):
                selected_cats = [cat_lookup[c] for c, _ in selected]
                penalty = selected_cats.count(cat_lookup[cid]) * diversity
                mmr_score = score - penalty
                if mmr_score > best_val:
                    best_val, best_idx = mmr_score, i
            selected.append(remaining.pop(best_idx))

        return selected
