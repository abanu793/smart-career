# app.py
import streamlit as st
import pandas as pd
from matching_engine import recommend

st.set_page_config(page_title="AI Course Recommendation", layout="wide")
st.title("🎓 AI Course Recommendation Engine")

# -------------------------------
# Load Courses
# -------------------------------
courses = pd.read_csv("courses.csv")

# Build technical skills list
all_skills = set()
for tags in courses["skill_tags"]:
    if pd.notna(tags):
        all_skills.update([t.strip() for t in tags.split(",")])
all_skills = sorted(list(all_skills))

# Predefined soft skills
soft_skills_list = [
    "Communication",
    "Leadership",
    "Teamwork",
    "Problem Solving",
    "Creativity",
]

# Education & Majors
education_levels = ["High School", "BSc", "B.Tech", "MSc", "MBA"]
majors = ["Computer Science", "IT", "Electronics", "Mathematics", "Business"]

# -------------------------------
# Sample User Profiles
# -------------------------------
sample_profiles = {
    "Beginner - High School": {
        "education": "High School",
        "major": "Mathematics",
        "technical_skills": [],
        "soft_skills": ["Communication"],
        "interests": "Programming basics",
        "career_goals": "Learn Python and Data Science basics",
    },
    "Intermediate - BSc CS": {
        "education": "BSc",
        "major": "Computer Science",
        "technical_skills": ["Python", "SQL"],
        "soft_skills": ["Teamwork", "Problem Solving"],
        "interests": "Machine Learning, AI",
        "career_goals": "Become ML Engineer",
    },
    "Advanced - MSc Data Science": {
        "education": "MSc",
        "major": "Computer Science",
        "technical_skills": ["Python", "Machine Learning", "TensorFlow"],
        "soft_skills": ["Leadership", "Problem Solving"],
        "interests": "Deep Learning, NLP",
        "career_goals": "Data Scientist / AI Researcher",
    },
    "Professional - MBA": {
        "education": "MBA",
        "major": "Business",
        "technical_skills": ["Excel", "Power BI", "SQL"],
        "soft_skills": ["Leadership", "Teamwork"],
        "interests": "Business Analytics",
        "career_goals": "Business Analyst / Consultant",
    },
    "Cloud Enthusiast": {
        "education": "B.Tech",
        "major": "IT",
        "technical_skills": ["AWS", "Docker", "Kubernetes"],
        "soft_skills": ["Problem Solving"],
        "interests": "Cloud Computing, DevOps",
        "career_goals": "Cloud Engineer / DevOps Specialist",
    },
}

# -------------------------------
# User Form
# -------------------------------
st.subheader("🧑‍🎓 User Profile")
profile_option = st.selectbox(
    "Choose a Sample Profile (or select 'Manual')",
    ["Manual"] + list(sample_profiles.keys()),
)

results = None  # initialize results

if profile_option == "Manual":
    col1, col2 = st.columns(2)
    with col1:
        education = st.selectbox("Education Level", education_levels)
        major = st.selectbox("Major", majors)
    with col2:
        tech_skills = st.multiselect("Technical Skills", all_skills)
        soft_skills = st.multiselect("Soft Skills", soft_skills_list)
    interests = st.text_input("Interests (optional)")
    career_goals = st.text_input("Career Goals (optional)")

    submitted = st.button("🔍 Get Recommendations")
    if submitted:
        profile = {
            "education": education,
            "major": major,
            "technical_skills": tech_skills,
            "soft_skills": soft_skills,
            "interests": interests,
            "career_goals": career_goals,
        }
        results = recommend(profile, top_k=15)
else:
    profile = sample_profiles[profile_option]
    st.write("Selected Sample Profile:", profile)
    results = recommend(profile, top_k=15)


# -------------------------------
# Helper: Convert results to DataFrame
# -------------------------------
def df_convert(data):
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Title": d["title"],
                "Provider": d["provider"],
                "Score": d["score"],
                "Level": d["level"],
                "Duration (weeks)": d["duration_weeks"],
                "Missing Prereqs": (
                    ", ".join(d["missing_prereqs"]) if d["missing_prereqs"] else "None"
                ),
                "Cost": d["cost"],
                "Type": d["type"],
                "Cost/Week": (
                    round(d["cost"] / d["duration_weeks"], 2)
                    if d["duration_weeks"] > 0
                    else d["cost"]
                ),
                "Explanation": d["explanation"],
                "Enroll": f'<a href="{d["link"]}" target="_blank">Enroll</a>',
            }
            for d in data
        ]
    )


# -------------------------------
# Display Recommendations if results exist
# -------------------------------
if results:
    # Short-term courses
    st.subheader("🟢 Short-Term Courses (≤ 12 weeks)")
    df_short = df_convert(results["short_term"])
    if df_short.empty:
        st.warning("No short-term courses matched your profile.")
    else:
        st.write(df_short.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Long-term courses
    st.subheader("🔵 Long-Term Courses (> 12 weeks or Advanced)")
    df_long = df_convert(results["long_term"])
    if df_long.empty:
        st.info("No long-term courses available for this profile.")
    else:
        st.write(df_long.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Score Visualization
    st.subheader("📊 Score Visualization")
    if not df_short.empty:
        st.dataframe(
            df_short.style.background_gradient(cmap="Greens", subset=["Score"])
        )
    if not df_long.empty:
        st.dataframe(df_long.style.background_gradient(cmap="Blues", subset=["Score"]))

    # JSON Output
    st.subheader("📄 JSON Output")
    st.json(results)
