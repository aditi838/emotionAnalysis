# Emotion and Sentiment Analysis

A Streamlit application that analyzes English text to detect emotions and sentiment using transformer-based language models. The app predicts one of seven core emotions, classifies overall sentiment, and visualizes prediction confidence with an interactive chart.

---

## What it does

- Classifies text into one of **7 core emotions** using DistilRoBERTa
- Detects **sentiment** (positive / negative / neutral) using a RoBERTa-based sentiment model
- Displays an **interactive confidence score bar chart** using Plotly
- Accepts both **typed text** and **uploaded `.txt` files**

### Supported emotion classes

- Anger
- Disgust
- Fear
- Joy
- Neutral
- Sadness
- Surprise

---

## How it works

```
User Input (text or .txt file)
        │
        ▼
DistilRoBERTa Emotion Model
        │
        ├──────────────► Emotion prediction (7 classes)
        │
        ▼
RoBERTa Sentiment Model
        │
        ├──────────────► Positive / Neutral / Negative
        │
        ▼
Interactive Plotly confidence chart
```

**Emotion model:** `j-hartmann/emotion-english-distilroberta-base` — DistilRoBERTa fine-tuned across six emotion datasets to classify seven core emotions.

**Sentiment model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` — RoBERTa model for contextual sentiment classification.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | UI and application framework |
| Hugging Face Transformers | Model loading and inference |
| PyTorch | Transformer inference backend |
| Plotly | Interactive confidence visualization |
| Pandas | Data formatting and visualization |
| DistilRoBERTa | Emotion classification |
| RoBERTa | Sentiment classification |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/aditi838/emotionAnalysis.git
cd emotionAnalysis
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch (`torch~=2.5.1`) is a large dependency and may take a few minutes to install.

### 4. Run the application

```bash
streamlit run main.py
```

The app will be available at:

```
http://localhost:8501
```

---

## Notes

- Both transformer models are downloaded from Hugging Face during the first run and cached for future use.
- The application runs entirely locally after the models have been downloaded.
- No API keys are required.

---

## Acknowledgements

- `j-hartmann/emotion-english-distilroberta-base` — Emotion classification model
- `cardiffnlp/twitter-roberta-base-sentiment-latest` — Sentiment classification model
- Hugging Face Transformers
- Streamlit
