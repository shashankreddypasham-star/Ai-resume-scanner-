import streamlit as st
import PyPDF2
import docx
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

# ------------------ FUNCTIONS ------------------

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

def calculate_similarity(job_desc, resumes):
    documents = [job_desc] + resumes
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return scores

# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("🤖 AI Resume Screener Web Application")
st.write("Upload resumes and match them with a job description using AI & NLP")

st.markdown("---")

# Job Description Input
st.subheader("📄 Job Description")
job_description = st.text_area("Enter Job Description", height=200)

# Resume Upload
st.subheader("📂 Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload PDF or DOCX resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# Process Button
if st.button("🔍 Screen Resumes"):

    if not job_description or not uploaded_files:
        st.error("Please upload resumes and enter a job description")
    else:
        cleaned_job_desc = clean_text(job_description)
        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            else:
                text = extract_text_from_docx(file)

            cleaned_text = clean_text(text)
            resume_texts.append(cleaned_text)
            resume_names.append(file.name)

        scores = calculate_similarity(cleaned_job_desc, resume_texts)

        results = list(zip(resume_names, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        st.markdown("---")
        st.subheader("📊 Resume Matching Results")

        for name, score in results:
            st.write(f"**{name}**")
            st.progress(int(score * 100))
            st.write(f"Match Score: **{round(score * 100, 2)}%**")
            st.markdown("---")

st.markdown("### ✅ Technologies Used")
st.markdown("""
- Streamlit  
- Python  
- NLP (TF-IDF)  
- Machine Learning  
- Cosine Similarity  
""")

st.markdown("### 🚀 Project By: AI Resume Screener using Streamlit")
