# matching_engine.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load courses
courses = pd.read_csv("courses.csv")

# Combine text for embeddings
courses["combined_text"] = courses.apply(
    lambda x: f"{x['title']} {x['skill_tags']} {x['description']}", axis=1
)
course_embeddings = model.encode(
    courses["combined_text"].tolist(), show_progress_bar=False
)


def vectorize_profile(profile):
    """Create embedding for user profile"""
    text = (
        f"Education: {profile['education']}. "
        f"Major: {profile['major']}. "
        f"Technical Skills: {', '.join(profile['technical_skills'])}. "
        f"Soft Skills: {', '.join(profile['soft_skills'])}. "
        f"Interests: {profile['interests']}. "
        f"Career Goals: {profile['career_goals']}."
    )
    return model.encode([text])[0]


def calculate_skill_match(user_skills, course_skills):
    """Returns skill match ratio (0–1)"""
    course_skills = [s.strip().lower() for s in course_skills.split(",") if s.strip()]
    user_skills = [s.lower() for s in user_skills]
    if not course_skills:
        return 0
    matches = set(user_skills).intersection(set(course_skills))
    return len(matches) / len(course_skills)


def recommend(profile, top_k=15):
    user_vec = vectorize_profile(profile)
    results = []

    for idx, row in courses.iterrows():
        course_vec = course_embeddings[idx]

        # Cosine similarity - map -1..1 -> 0..100
        cos_sim = float(cosine_similarity([user_vec], [course_vec])[0][0])
        score_cosine = (cos_sim + 1) * 50

        # Skill match score
        skill_match = calculate_skill_match(
            profile["technical_skills"], row["skill_tags"]
        )
        skill_match_score = skill_match * 100

        # Missing prerequisites
        missing_prereqs = []
        prereqs = []  # initialize to avoid UnboundLocalError
        if pd.notna(row["prerequisites"]):
            prereqs = [
                p.strip().lower() for p in row["prerequisites"].split(",") if p.strip()
            ]
            user_lower = [u.lower() for u in profile["technical_skills"]]
            missing_prereqs = [p for p in prereqs if p not in user_lower]

        # Block advanced courses if user lacks all prereqs
        if (
            row["level"].lower() == "advanced"
            and len(prereqs) > 0
            and len(missing_prereqs) == len(prereqs)
        ):
            final_score = 0
        else:
            final_score = (
                0.6 * score_cosine
                + 0.3 * skill_match_score
                - 0.1 * (len(missing_prereqs) > 0) * 100
            )
            final_score = max(min(final_score, 100), 0)

        final_score = int(round(final_score))  # integer 0–100

        # Justification / Explanation
        explanation = (
            f"Skill match: {int(skill_match*100)}%. "
            f"{'Missing prereqs: ' + ', '.join(missing_prereqs) if missing_prereqs else 'No prereqs missing.'} "
            f"Recommended preparation: {', '.join(missing_prereqs) if missing_prereqs else 'None'}."
        )

        # Cost per week
        cost_per_week = (
            round(row["cost"] / row["duration_weeks"], 2)
            if pd.notna(row.get("cost")) and row["duration_weeks"] > 0
            else "N/A"
        )

        results.append(
            {
                "title": row["title"],
                "provider": row["provider"],
                "score": final_score,
                "level": row["level"],
                "duration_weeks": row["duration_weeks"],
                "link": row["link"],
                "missing_prereqs": missing_prereqs,
                "explanation": explanation,
                "cost": row.get("cost", "N/A"),
                "type": row.get("type", "N/A"),
                "cost_per_week": cost_per_week,
            }
        )

    # Sort by score
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    # Stepwise: short-term <=12 weeks, long-term >12 weeks or Advanced
    short_term = [
        r
        for r in sorted_results
        if r["duration_weeks"] <= 12 and r["level"].lower() != "advanced"
    ]
    long_term = [
        r
        for r in sorted_results
        if r["duration_weeks"] > 12 or r["level"].lower() == "advanced"
    ]

    return {"short_term": short_term, "long_term": long_term}
