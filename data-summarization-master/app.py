from flask import Flask, render_template, request, jsonify
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

app = Flask(__name__)

# Hugging Face Model Setup (loaded once)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Executor for running blocking code asynchronously
executor = ThreadPoolExecutor(max_workers=4)

# Function to preprocess text before summarization
def preprocess_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r"[^a-zA-Z0-9\s.,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize sentences (optional for Hugging Face model)
    sentences = sent_tokenize(text)

    # Remove stopwords (optional)
    stop_words = set(stopwords.words("english"))
    processed_sentences = []
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        processed_sentences.append(" ".join(filtered_words))

    return " ".join(processed_sentences)

# Function to summarize text using the Hugging Face pipeline
def summarize_text(text):
    try:
        summary = summarizer(text, min_length=60, max_length=100)
        return summary[0]["summary_text"] if summary else "Error: Summary could not be generated."
    except Exception as e:
        return f"Error: {e}"

# Function to run summarization asynchronously using ThreadPoolExecutor
def run_summarization_async(text):
    return executor.submit(summarize_text, text)

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    summary = None

    if request.method == "POST":
        text_to_summarize = request.form.get("text", "").strip()

        if not text_to_summarize:  # Prevent empty input errors
            return render_template("index.html", summary="Error: No text provided.")

        preprocessed_text = preprocess_text(text_to_summarize)  # Apply text processing

        # Run the summarization asynchronously
        future = run_summarization_async(preprocessed_text)

        # Wait for the result and get the summary
        summary = future.result()

        # Print debug info in console
        print(f"Input Text: {text_to_summarize}")
        print(f"Preprocessed Text: {preprocessed_text}")
        print(f"Summary: {summary}")

    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
