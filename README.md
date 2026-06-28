# Emotion and Sentiment Analysis

A Streamlit app that detects emotions and sentiment from text input using a fine-tuned RoBERTa model trained on Google's GoEmotions dataset. Results are displayed as an interactive probability chart across all 28 emotion categories.

---

## What it does

- Classifies text into one of **28 emotion categories** using `roberta_goEmotion`
- Determines **sentiment** (positive / negative / neutral) using TextBlob polarity scoring
- Displays a **confidence score bar chart** across all emotions using Plotly
- Accepts both **typed text** and **uploaded `.txt` files**

### The 28 emotion classes

admiration · amusement · anger · annoyance · approval · caring · confusion · curiosity · desire · disappointment · disapproval · disgust · embarrassment · excitement · fear · gratitude · grief · joy · love · nervousness · optimism · pride · realization · relief · remorse · sadness · surprise · neutral

---

## How it works

```
User Input (text or .txt file)
        │
        ▼
RoBERTa (roberta_goEmotion)     TextBlob
        │                           │
        ▼                           ▼
28-class emotion + probabilities   Sentiment polarity score
        │                           │
        └───────────┬───────────────┘
                    ▼
         Interactive Plotly bar chart
```

**Emotion model:** `bsingh/roberta_goEmotion` — RoBERTa fine-tuned on [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions), Google's dataset of 58k Reddit comments labelled across 27 emotions + neutral.

**Sentiment:** TextBlob rule-based polarity scoring — positive (> 0), negative (< 0), neutral (= 0).

---

## Tech stack

| Tool | Purpose |
|---|---|
| Streamlit | UI and app framework |
| Transformers (HuggingFace) | RoBERTa model loading and inference |
| PyTorch | Model inference backend |
| TextBlob | Sentiment polarity scoring |
| Plotly | Interactive emotion probability chart |
| Pandas | Data formatting for chart |

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

> **Note:** PyTorch (`torch~=2.5.1`) is a large download (~2GB). This may take a few minutes.

### 4. Run the app

```bash
streamlit run main.py
```

The app opens at `http://localhost:8501`.

---

## Notes

- The RoBERTa model is downloaded from HuggingFace on first run (~500MB). Subsequent runs load it from cache.
- `pydub` and `ffmpeg` are listed as dependencies for audio processing — voice input is not currently implemented in the UI.
- No API keys required — the model runs entirely locally.

---

## Acknowledgements

- [GoEmotions dataset](https://huggingface.co/datasets/google-research-datasets/go_emotions) — Google Research
- [roberta_goEmotion](https://huggingface.co/bsingh/roberta_goEmotion) — bsingh on HuggingFace
- [TextBlob](https://textblob.readthedocs.io/)
