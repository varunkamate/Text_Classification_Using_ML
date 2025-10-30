import streamlit as st
import re
import string
import joblib
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are available
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ---- Preprocessing Function ----
def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)                 # remove HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)        # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)              # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()           # normalize whitespace
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ---- Load model and vectorizer ----
@st.cache_resource
def load_resources():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# ---- Streamlit App ----
def main():
    st.set_page_config(page_title="Text Classification", layout="centered")
    st.title("📝 Text Classification (Streamlit)")
    st.write("Enter text and click Predict. The model will classify it based on your trained ML model.")

    model, vectorizer = load_resources()

    user_input = st.text_area("Enter text below:", height=200)

    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter some text to classify.")
        else:
            cleaned = preprocess(user_input)
            X = vectorizer.transform([cleaned])
            pred = model.predict(X)[0]

            st.success(f"**Predicted Label:** {pred}")
            st.write("---")
            st.write("**Original text:**", user_input)
            st.write("**Preprocessed text:**", cleaned)

if __name__ == "__main__":
    main()
