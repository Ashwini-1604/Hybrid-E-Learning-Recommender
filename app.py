import pandas as pd
import streamlit as st

from recommender.collaborative import CollaborativeRecommender
from recommender.content_based import ContentBasedRecommender
from recommender.hybrid import HybridRecommender
from recommender.local_reranker import LocalReranker

st.set_page_config(page_title="Hybrid E-Learning Recommender", page_icon="🎓", layout="wide")


@st.cache_data
def load_data():
    courses = pd.read_csv("data/courses.csv")
    interactions = pd.read_csv("data/interactions.csv")
    return courses, interactions


@st.cache_resource
def build_models(_courses, _interactions):
    cf = CollaborativeRecommender(n_factors=20).fit(_interactions)
    cb = ContentBasedRecommender().fit(_courses)
    hybrid = HybridRecommender(cf, cb)
    reranker = LocalReranker(cb)
    return cf, cb, hybrid, reranker


def main():
    st.title("🎓 Hybrid E-Learning Recommender")
    st.caption(
        "Collaborative filtering × content-based filtering × LLM re-ranking "
        "based on your career goal."
    )

    courses_df, interactions_df = load_data()
    cf_model, cb_model, hybrid_model, reranker = build_models(courses_df, interactions_df)

    with st.sidebar:
        st.header("Learner Profile")
        user_ids = sorted(interactions_df["user_id"].unique().tolist())
        user_mode = st.radio("User", ["Existing learner", "New learner (cold-start)"])

        if user_mode == "Existing learner":
            user_id = st.selectbox("Pick a learner ID", user_ids)
            taken = interactions_df[interactions_df["user_id"] == user_id].merge(
                courses_df, on="course_id", suffixes=("_given", "_course")
            )
            st.markdown("**Course history:**")
            st.dataframe(
                taken[["title", "category", "completed", "rating_given"]]
                .rename(columns={"rating_given": "your_rating"}),
                hide_index=True, use_container_width=True
            )
        else:
            user_id = 999999  # unseen id -> triggers cold-start path in CF model

        goal_text = st.text_area(
            "What's your goal right now?",
            placeholder="e.g. I want to get into backend development but I'm weak at databases",
        )

        top_n = st.slider("Number of recommendations", 3, 10, 6)
        use_llm = st.checkbox("Use goal-aware re-ranking", value=True)
        go = st.button("Get Recommendations", type="primary", use_container_width=True)

    mode = "semantic (sentence-transformers)" if cb_model.use_semantic else "TF-IDF fallback"
    st.caption(f"Content matching mode: **{mode}** · 100% free, runs locally, no API key needed.")

    if go:
        with st.spinner("Crunching collaborative + content signals..."):
            candidates = hybrid_model.recommend(
                user_id, courses_df, goal_text=goal_text, top_n=top_n * 2
            )

        final_rows = None
        if use_llm and goal_text.strip():
            with st.spinner("Asking the LLM to re-rank for your specific goal..."):
                final_rows = reranker.rerank(goal_text, candidates, top_n=top_n)

        st.subheader("Recommended for you")

        if final_rows:
            for row in final_rows:
                render_course_card(row, show_reason=True)
        else:
            for _, row in candidates.head(top_n).iterrows():
                render_course_card(row.to_dict(), show_reason=False)

        with st.expander("🔍 See the scoring breakdown (for your report/demo)"):
            st.dataframe(
                candidates.head(top_n)[
                    ["title", "hybrid_score", "cf_score", "cb_score", "goal_score"]
                ],
                hide_index=True, use_container_width=True
            )
            st.caption(
                "cf_score = collaborative filtering (peer behavior) · "
                "cb_score = content similarity to your history · "
                "goal_score = similarity to your stated goal text"
            )


def render_course_card(row, show_reason):
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"**{row['title']}**")
            st.caption(f"{row['category']} · {row['difficulty']} · {row['duration_hours']}h · ⭐ {row['rating']}")
            st.write(row["description"])
            if show_reason and row.get("llm_reason"):
                st.success(f"💡 Why this fits your goal: {row['llm_reason']}")
        with c2:
            st.metric("Match", f"{row.get('hybrid_score', 0):.0%}" if row.get('hybrid_score') is not None else "—")


if __name__ == "__main__":
    main()
