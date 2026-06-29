import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

MAX_TOKENS = 512

EMOTION_EMOJI = {
    "anger": "😡",
    "disgust": "🤢",
    "fear": "😱",
    "joy": "😊",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😲",
}

SENTIMENT_EMOJI = {
    "Positive": "😄",
    "Negative": "☹️",
    "Neutral": "😐",
}


# ─────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():

    # Emotion model
    emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_MODEL
    )
    emotion_model.eval()

    emotion_labels = [
        emotion_model.config.id2label[i].lower()
        for i in range(emotion_model.config.num_labels)
    ]

    # Sentiment model
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL
    )
    sentiment_model.eval()

    return (
        emotion_tokenizer,
        emotion_model,
        emotion_labels,
        sentiment_tokenizer,
        sentiment_model,
    )


# ─────────────────────────────────────────────────────────────
# Emotion analysis
# ─────────────────────────────────────────────────────────────

def analyze_emotion(text, tokenizer, model, labels):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0].tolist()

    top_idx = torch.argmax(outputs.logits, dim=1).item()

    return labels[top_idx], probabilities


# ─────────────────────────────────────────────────────────────
# Sentiment analysis
# ─────────────────────────────────────────────────────────────

def get_sentiment(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    label = model.config.id2label[prediction].lower()

    if "negative" in label:
        return "Negative"

    elif "neutral" in label:
        return "Neutral"

    else:
        return "Positive"


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def estimate_tokens(text):
    return len(text.split())


# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────

def display_results(emotion, probabilities, sentiment, labels):

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Detected Emotion",
            f"{emotion.capitalize()} {EMOTION_EMOJI.get(emotion,'')}",
        )

    with col2:
        st.metric(
            "Sentiment",
            f"{sentiment} {SENTIMENT_EMOJI[sentiment]}",
        )

    emotion_df = pd.DataFrame(
        {
            "Emotion": [x.capitalize() for x in labels],
            "Probability": probabilities,
        }
    )

    emotion_df = emotion_df.sort_values(
        by="Probability",
        ascending=False,
    )

    fig = px.bar(
        emotion_df,
        x="Emotion",
        y="Probability",
        title="Emotion Confidence Scores",
        color="Probability",
        color_continuous_scale="Blues",
        labels={"Probability": "Confidence Score"},
    )

    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_tickangle=0,
    )

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Emotion & Sentiment Analysis",
    layout="wide",
)

st.title("🧠 Emotion and Sentiment Analysis")

st.caption(
    "Powered by DistilRoBERTa (Emotion) + CardiffNLP RoBERTa (Sentiment)"
)

(
    emotion_tokenizer,
    emotion_model,
    EMOTION_LABELS,
    sentiment_tokenizer,
    sentiment_model,
) = load_models()

tab1, tab2 = st.tabs(
    [
        "✍️ Enter Text",
        "📄 Upload File",
    ]
)

# ============================================================
# Text input
# ============================================================

with tab1:

    user_input = st.text_area(
        "Enter text to analyse:",
        height=150,
    )

    if user_input.strip():

        token_estimate = estimate_tokens(user_input)

        if token_estimate > MAX_TOKENS:

            st.warning(
                f"Your text is approximately {token_estimate} words. "
                f"The model processes up to {MAX_TOKENS} tokens, "
                "so longer text will be truncated."
            )

    if st.button("Analyse", key="text"):

        if user_input.strip():

            with st.spinner("Analysing..."):

                emotion, probs = analyze_emotion(
                    user_input,
                    emotion_tokenizer,
                    emotion_model,
                    EMOTION_LABELS,
                )

                sentiment = get_sentiment(
                    user_input,
                    sentiment_tokenizer,
                    sentiment_model,
                )

            display_results(
                emotion,
                probs,
                sentiment,
                EMOTION_LABELS,
            )

        else:
            st.warning("Please enter some text.")


# ============================================================
# File Upload
# ============================================================

with tab2:

    uploaded_file = st.file_uploader(
        "Upload a .txt file",
        type=["txt"],
    )

    if uploaded_file is not None:

        file_content = uploaded_file.read().decode("utf-8")

        with st.expander("View file content"):
            st.write(file_content)

        token_estimate = estimate_tokens(file_content)

        if token_estimate > MAX_TOKENS:

            st.warning(
                f"File is approximately {token_estimate} words. "
                f"The model processes up to {MAX_TOKENS} tokens, "
                "so longer text will be truncated."
            )

        if st.button("Analyse File", key="file"):

            with st.spinner("Analysing..."):

                emotion, probs = analyze_emotion(
                    file_content,
                    emotion_tokenizer,
                    emotion_model,
                    EMOTION_LABELS,
                )

                sentiment = get_sentiment(
                    file_content,
                    sentiment_tokenizer,
                    sentiment_model,
                )

            display_results(
                emotion,
                probs,
                sentiment,
                EMOTION_LABELS,
            )
