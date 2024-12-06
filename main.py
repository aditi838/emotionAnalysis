import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import pandas as pd
import plotly.express as px

# Load pre-trained model and tokenizer
MODEL_NAME = "bsingh/roberta_goEmotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Emotion labels based on GoEmotions dataset
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude",
    "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Function to analyze emotion with probabilities
def analyze_emotion_with_probs(user_text):
    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0]
    max_prob_idx = probabilities.index(max(probabilities))
    return emotion_labels[max_prob_idx], probabilities

# Function to determine sentiment
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

# Streamlit UI setup
st.set_page_config(page_title="Emotion Detection", layout="wide")

st.title("Emotion and Sentiment Analysis")

# Function to display results with an interactive chart
def display_results(user_input, emotion, probabilities, sentiment):
    # Emojis for visualization
    emotion_emoji = {"joy": "😊", "anger": "😡", "fear": "😱", "neutral": "😐", "sadness": "😢"}
    sentiment_emoji = {"Positive": "😄", "Negative": "☹️", "Neutral": "😐"}

    # Display emotion and sentiment with emojis
    st.markdown(f"**Emotion:** {emotion} {emotion_emoji.get(emotion, '')}")
    st.markdown(f"**Sentiment:** {sentiment} {sentiment_emoji[sentiment]}")

    # Create interactive bar chart
    emotion_data = pd.DataFrame({
        "Emotion": emotion_labels,
        "Probability": probabilities
    })

    fig = px.bar(emotion_data, x="Emotion", y="Probability",
                 title="Emotion Probabilities",
                 labels={"Probability": "Confidence Score"})
    st.plotly_chart(fig, use_container_width=True)

# Text Analysis Section
user_input = st.text_area("Enter text here:")

uploaded_text_file = st.file_uploader("Upload a text file for analysis", type=["txt"])
if uploaded_text_file is not None:
    file_content = uploaded_text_file.read().decode("utf-8")
    st.write("**File Content:**")
    st.write(file_content)

    # Automatically process uploaded file
    emotion, probabilities = analyze_emotion_with_probs(file_content)
    sentiment = get_sentiment(file_content)
    display_results(file_content, emotion, probabilities, sentiment)

if st.button("Analyze Text"):
    combined_text = user_input.strip()
    if combined_text:
        # Analyze emotion and sentiment
        emotion, probabilities = analyze_emotion_with_probs(combined_text)
        sentiment = get_sentiment(combined_text)
        # Display results
        display_results(combined_text, emotion, probabilities, sentiment)
    else:
        st.warning("Please enter some text.")
