import streamlit as st
import re
import nltk
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

st.set_page_config(
    page_title="Skill Assessment Profile",
    layout="centered"
)

st.title("Skill Assessment Profile")

# ---------------- ROLE SKILLS ----------------

ROLE_SKILLS = {

"Data Analyst":[
"sql","excel","python","tableau",
"power bi","data visualization","statistics","data dashboarding"
],

"Data Scientist":[
"python","statistics","machine learning",
"pandas","numpy","sql","scikit learn"
],

"Machine Learning Engineer":[
"python","machine learning","deep learning",
"tensorflow","pytorch","deployment","docker"
],

"Software Engineer":[
"python","java","c++","data structures",
"algorithms","oop","git"
],

"Web Developer":[
"html","css","javascript","react",
"node","frontend","backend"
],

"Accountant":[
"accounting","tally","gst","taxation",
"financial statements","auditing","excel"
],

"Financial Analyst":[
"financial analysis","excel","valuation",
"budgeting","forecasting","sql"
],

"Business Analyst":[
"business analysis","requirements",
"sql","excel","power bi",
"data visualization","stakeholder"
]

}

# ---------------- FUNCTIONS ----------------

def extract_text(pdf):

    reader = PdfReader(pdf)

    text = ""

    for page in reader.pages:

        if page.extract_text():

            text += page.extract_text()

    return text.lower()


def clean_text(text):

    text = re.sub(r'[^a-zA-Z ]',' ',text)

    tokens = text.split()

    return " ".join(
        [t for t in tokens if t not in stop_words]
    )


def extract_name(text):

    lines = text.split("\n")

    for line in lines[:5]:

        if 1 < len(line.split()) <= 4:

            return line.title()

    return "Not clearly detected"


def estimate_experience(text):

    exp = re.findall(r'(\d+)\+?\s+years?',text)

    return f"{max(exp)} years" if exp else "Fresher / Not specified"


def extract_skills(text):

    all_skills = set(sum(ROLE_SKILLS.values(),[]))

    return sorted(
        [s for s in all_skills if s in text]
    )


def role_scores_algorithm(cleaned_text):

    scores = {}

    for role,skills in ROLE_SKILLS.items():

        matched = [s for s in skills if s in cleaned_text]

        scores[role] = round(
            (len(matched)/len(skills))*100,
            2
        )

    return scores


def train_ml_model():

    texts = [" ".join(skills)
             for skills in ROLE_SKILLS.values()]

    labels = list(ROLE_SKILLS.keys())

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)

    model.fit(X,labels)

    return model,vectorizer


# ---------------- FILE UPLOAD ----------------

st.subheader("Upload Resume")

resume_file = st.file_uploader(
"Upload your resume in PDF format",
type=["pdf"]
)

# ---------------- MAIN LOGIC ----------------

if resume_file:

    raw_text = extract_text(resume_file)

    cleaned = clean_text(raw_text)

    st.divider()

    st.subheader("Resume Summary")

    col1,col2 = st.columns(2)

    with col1:

        st.info(
        f"Candidate Name: {extract_name(raw_text)}"
        )

    with col2:

        st.info(
        f"Experience: {estimate_experience(raw_text)}"
        )

    skills_found = extract_skills(cleaned)

    st.write("Detected Technical Skills:")

    st.write(
    ", ".join(skills_found)
    if skills_found else
    "No relevant skills detected."
    )

    st.divider()

    st.subheader("Career Suitability Analysis")

    scores = role_scores_algorithm(cleaned)

    best_role = max(scores,key=scores.get)

    for role,score in scores.items():

        st.write(f"{role} → {score}% match")

    st.success(
    f"Recommended Role : {best_role}"
    )

    st.divider()

    st.subheader("Statistical Classification Confidence")

    model,vectorizer = train_ml_model()

    resume_vec = vectorizer.transform([cleaned])

    probabilities = model.predict_proba(resume_vec)[0]

    roles = model.classes_

    role_prob_pairs = list(zip(roles,probabilities))

    role_prob_pairs.sort(
        key=lambda x:x[1],
        reverse=True
    )

    top_role = role_prob_pairs[0][0]

    top_confidence = round(
        role_prob_pairs[0][1]*100,
        2
    )

    st.success(
    f"Predicted Role : {top_role} ({top_confidence}% confidence)"
    )

    st.write("Top Role Probabilities")

    for role,prob in role_prob_pairs[:3]:

        st.write(
        f"{role} → {round(prob*100,2)}%"
        )

    st.divider()

    st.subheader("Target Role Evaluation")

    selected_role = st.selectbox(
    "Select your desired career role:",
    list(ROLE_SKILLS.keys())
    )

    target_score = scores.get(selected_role,0)

    st.metric(
    "Skill Match Percentage",
    f"{target_score}%"
    )

    if target_score >= 75:

        st.success(
        "Excellent alignment with this role."
        )

    elif 50 <= target_score < 75:

        st.warning(
        "Moderate alignment. Skill enhancement recommended."
        )

    else:

        st.error(
        "Low alignment. Significant skill gap detected."
        )

    st.divider()

    st.subheader(
    "Skill Gap Analysis & Improvement Suggestions"
    )

    required_skills = ROLE_SKILLS[selected_role]

    missing_skills = [
    s for s in required_skills
    if s not in cleaned
    ]

    if missing_skills:

        st.write(
        "To improve suitability for this role:"
        )

        for skill in missing_skills:

            st.write(f"- {skill}")

    else:

        st.success(
        "All core skills present."
        )