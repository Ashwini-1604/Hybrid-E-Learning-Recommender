"""
Generates a realistic synthetic e-learning dataset so the whole project
runs end-to-end without needing to scrape Coursera/edX (which usually
requires paid APIs or fragile scraping).

Swap this out later with a real scraped/Kaggle dataset — just make sure
the output CSVs keep the same column names and the rest of the pipeline
will keep working unchanged.
"""

import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

CATEGORIES = {
    "Backend Development": [
        "REST APIs", "databases", "system design", "microservices", "SQL",
        "authentication", "caching", "server architecture", "Node.js", "Django"
    ],
    "Frontend Development": [
        "React", "CSS", "responsive design", "JavaScript", "UI components",
        "accessibility", "state management", "Vue", "TypeScript", "animations"
    ],
    "Data Science": [
        "pandas", "statistics", "data visualization", "regression", "EDA",
        "hypothesis testing", "NumPy", "data cleaning", "SQL for analytics"
    ],
    "Machine Learning": [
        "neural networks", "scikit-learn", "model evaluation", "feature engineering",
        "deep learning", "PyTorch", "transformers", "computer vision", "NLP"
    ],
    "Cloud & DevOps": [
        "AWS", "Docker", "Kubernetes", "CI/CD", "Terraform", "monitoring",
        "load balancing", "serverless", "networking basics"
    ],
    "Cybersecurity": [
        "penetration testing", "network security", "cryptography", "OWASP top 10",
        "threat modeling", "secure coding", "incident response"
    ],
    "Mobile Development": [
        "Flutter", "Kotlin", "Swift", "React Native", "mobile UI patterns",
        "app store deployment", "offline storage"
    ],
}

DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]

TITLE_TEMPLATES = [
    "Introduction to {skill}", "Mastering {skill}", "{skill} for Beginners",
    "Advanced {skill}", "{skill} in Practice", "Complete Guide to {skill}",
    "{skill} Bootcamp", "Hands-on {skill}", "{skill} Fundamentals",
    "Building Projects with {skill}"
]


def generate_courses(n_courses=120):
    rows = []
    course_id = 1
    for category, skills in CATEGORIES.items():
        n_per_cat = n_courses // len(CATEGORIES)
        for _ in range(n_per_cat):
            skill = random.choice(skills)
            template = random.choice(TITLE_TEMPLATES)
            title = template.format(skill=skill)
            difficulty = random.choice(DIFFICULTIES)
            related_skills = random.sample(skills, k=min(4, len(skills)))
            description = (
                f"This course on {title} covers {', '.join(related_skills)}. "
                f"Designed for {difficulty.lower()} learners aiming to build "
                f"practical, job-ready skills in {category.lower()}. "
                f"Includes hands-on projects and real-world case studies."
            )
            rows.append({
                "course_id": course_id,
                "title": title,
                "category": category,
                "difficulty": difficulty,
                "skills": ", ".join(related_skills),
                "description": description,
                "duration_hours": random.choice([4, 6, 8, 10, 15, 20, 30]),
                "rating": round(random.uniform(3.5, 5.0), 1),
            })
            course_id += 1
    return pd.DataFrame(rows)


def generate_interactions(courses_df, n_users=300, min_courses=3, max_courses=12):
    """
    Simulates learners who tend to stick to 1-2 categories (like real learners),
    with some randomness / cross-category exploration mixed in.
    """
    rows = []
    categories = courses_df["category"].unique().tolist()

    for user_id in range(1, n_users + 1):
        primary_cats = random.sample(categories, k=random.choice([1, 2]))
        n_taken = random.randint(min_courses, max_courses)

        pool = courses_df[courses_df["category"].isin(primary_cats)]
        # mix in a few courses from outside their main interest (exploration)
        n_primary = int(n_taken * 0.8)
        n_explore = n_taken - n_primary

        primary_courses = pool.sample(n=min(n_primary, len(pool)), replace=False)
        explore_courses = courses_df.sample(n=n_explore)

        taken = pd.concat([primary_courses, explore_courses]).drop_duplicates("course_id")

        for _, course in taken.iterrows():
            completed = random.random() > 0.2
            rating = np.clip(np.random.normal(4.0, 0.7), 1, 5) if completed else np.nan
            rows.append({
                "user_id": user_id,
                "course_id": course["course_id"],
                "completed": completed,
                "rating": round(rating, 1) if not np.isnan(rating) else None,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    courses = generate_courses()
    interactions = generate_interactions(courses)

    courses.to_csv("data/courses.csv", index=False)
    interactions.to_csv("data/interactions.csv", index=False)

    print(f"Generated {len(courses)} courses and {len(interactions)} interactions.")
    print("Saved to data/courses.csv and data/interactions.csv")
