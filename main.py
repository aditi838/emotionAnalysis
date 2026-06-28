import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import pandas as pd
import plotly.express as px

# ── Constants ────────────────────────────────────────────────
MODEL_NAME = "bsingh/roberta_goEmotion"

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude",
    "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

EMOTION_EMOJI = {
    "joy": "😊", "anger": "😡", "fear": "😱", "neutral": "😐",
    "sadness": "😢", "love": "❤️", "excitement": "🤩", "gratitude": "🙏",
    "disgust": "🤢", "surprise": "😲", "optimism": "✨", "grief": "😭"
}

SENTIMENT_EMOJI = {"Positive": "😄", "Negative": "☹️", "Neutral": "😐"}

MAX_TOKENS = 512


# ── Model loading (cached — loads once, reused across all reruns) ──
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


# ── Analysis functions ───────────────────────────────────────
def analyze_emotion(text: str, tokenizer, model):
    """Returns (top_emotion, probabilities_list)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0]
    top_idx = probabilities.index(max(probabilities))
    return EMOTION_LABELS[top_idx], probabilities


def get_sentiment(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"


def estimate_tokens(text: str) -> int:
    """Rough token estimate — actual tokenisation may differ slightly."""
    return len(text.split())


# ── Display function ─────────────────────────────────────────
def display_results(emotion: str, probabilities: list, sentiment: str):
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Detected Emotion",
            value=f"{emotion.capitalize()} {EMOTION_EMOJI.get(emotion, '')}"
        )
    with col2:
        st.metric(
            label="Sentiment",
            value=f"{sentiment} {SENTIMENT_EMOJI[sentiment]}"
        )

    emotion_data = pd.DataFrame({
        "Emotion": EMOTION_LABELS,
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)

    fig = px.bar(
        emotion_data,
        x="Emotion",
        y="Probability",
        title="Emotion Confidence Scores",
        labels={"Probability": "Confidence Score"},
        color="Probability",
        color_continuous_scale="Blues"
    )
    fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ── App ──────────────────────────────────────────────────────
st.set_page_config(page_title="Emotion & Sentiment Analysis", layout="wide")
st.title("🧠 Emotion and Sentiment Analysis")
st.caption("Powered by RoBERTa fine-tuned on GoEmotions · 28 emotion classes")

tokenizer, model = load_model()

tab1, tab2 = st.tabs(["✍️ Enter Text", "📄 Upload File"])

# ── Tab 1: Text input ────────────────────────────────────────
with tab1:
    user_input = st.text_area("Enter text to analyse:", height=150)

    if user_input.strip():
        token_estimate = estimate_tokens(user_input)
        if token_estimate > MAX_TOKENS:
            st.warning(
                f"Your text is approximately {token_estimate} words. "
                f"RoBERTa processes up to {MAX_TOKENS} tokens — longer text will be truncated."
            )

    if st.button("Analyse", key="btn_text"):
        text = user_input.strip()
        if text:
            with st.spinner("Analysing..."):
                emotion, probs = analyze_emotion(text, tokenizer, model)
                sentiment = get_sentiment(text)
            display_results(emotion, probs, sentiment)
        else:
            st.warning("Please enter some text.")

# ── Tab 2: File upload ───────────────────────────────────────
with tab2:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")

        with st.expander("View file content"):
            st.write(file_content)

        token_estimate = estimate_tokens(file_content)
        if token_estimate > MAX_TOKENS:
            st.warning(
                f"File is approximately {token_estimate} words. "
                f"RoBERTa processes up to {MAX_TOKENS} tokens — text will be truncated."
            )

        if st.button("Analyse File", key="btn_file"):
            with st.spinner("Analysing..."):
                emotion, probs = analyze_emotion(file_content, tokenizer, model)
                sentiment = get_sentiment(file_content)
            display_results(emotion, probs, sentiment)
