"""
Local, free re-ranking layer — replaces the paid Claude API re-ranker.

Instead of calling a paid LLM to write justifications, this uses the same
semantic embeddings from ContentBasedRecommender to find which specific
skills in each candidate course best match the learner's stated goal, then
builds a natural-language explanation from a template. It's deterministic,
costs nothing, needs no API key, and never hallucinates a reason that isn't
actually grounded in the course data.
"""


class LocalReranker:
    def __init__(self, content_model):
        self.content_model = content_model  # reuses the same embedding model
        self.enabled = True  # always available — no external dependency

    def rerank(self, goal_text: str, candidates_df, top_n=5):
        if not goal_text.strip():
            return None  # caller falls back to pure hybrid order

        rows = candidates_df.to_dict(orient="records")
        enriched = []
        for row in rows:
            matched_skills = self.content_model.top_matching_skills(
                goal_text, row.get("skills", ""), top_k=3
            )
            enriched.append((row, matched_skills))

        # Sort by: number of matched skills first, then existing hybrid score
        enriched.sort(key=lambda x: (len(x[1]), x[0].get("hybrid_score", 0)), reverse=True)

        results = []
        for row, matched_skills in enriched[:top_n]:
            row = dict(row)
            if matched_skills:
                row["llm_reason"] = (
                    f"Directly covers {', '.join(matched_skills)}, which line up with your goal."
                )
            else:
                row["llm_reason"] = (
                    "Ranked highly by the hybrid model based on similar learners and course content."
                )
            results.append(row)
        return results
