import streamlit as st
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
from pydub import AudioSegment
import io
import pandas as pd
import plotly.express as px
import re

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


# Initialize recognizer for voice input
recognizer = sr.Recognizer()

# Streamlit UI setup
st.set_page_config(page_title="Emotion Detection", layout="wide")

# Sidebar menu
page = st.sidebar.selectbox("Select Task", ["Text Analysis", "Voice Analysis"])


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


# Task 1: Text Analysis
if page == "Text Analysis":
    st.title("Text Analysis")
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

# Task 2: Voice Analysis
elif page == "Voice Analysis":
    st.title("Voice Emotion Analysis")

    # Option 1: Microphone Input
    if st.button("Start Recording"):
        with st.spinner("Listening..."):
            try:
                # Use the microphone to capture audio
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)
                    # Recognize the speech using Google's speech-to-text API
                    user_text = recognizer.recognize_google(audio)
                    st.write(f"**Transcribed Text:** {user_text}")

                    # Perform emotion analysis and sentiment detection
                    emotion, probabilities = analyze_emotion_with_probs(user_text)
                    sentiment = get_sentiment(user_text)
                    # Display results
                    display_results(user_text, emotion, probabilities, sentiment)

            except sr.UnknownValueError:
                st.error("Sorry, I couldn't understand the audio.")
            except sr.RequestError as e:
                st.error(f"Error with speech recognition service: {e}")

    # Option 2: Audio File Upload
    uploaded_audio = st.file_uploader("Upload an audio file (e.g., .wav, .mp3)", type=["wav", "mp3"])
    if uploaded_audio:
        with st.spinner("Processing the uploaded audio..."):
            try:
                # Load audio file
                audio_bytes = uploaded_audio.read()
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

                # Convert to .wav if not already
                if uploaded_audio.type != 'audio/wav':
                    audio = audio.set_channels(1).set_frame_rate(16000)
                    wav_buffer = io.BytesIO()
                    audio.export(wav_buffer, format="wav")
                    wav_buffer.seek(0)
                    audio_file = wav_buffer
                else:
                    audio_file = io.BytesIO(audio_bytes)

                # Recognize speech from the audio file using SpeechRecognition
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)
                    user_text = recognizer.recognize_google(audio_data)
                    st.write(f"**Transcribed Text:** {user_text}")

                    # Perform emotion analysis and sentiment detection
                    emotion, probabilities = analyze_emotion_with_probs(user_text)
                    sentiment = get_sentiment(user_text)
                    # Display results
                    display_results(user_text, emotion, probabilities, sentiment)

            except Exception as e:
                st.error(f"Error processing the audio file: {e}")
