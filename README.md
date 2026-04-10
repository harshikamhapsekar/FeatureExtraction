# 📊 Feature Extraction and Communication Analytics System


A Python-based pipeline that extracts and analyzes communication quality features from YouTube tutorial videos using transcript data and video metadata. Computes measurable linguistic and engagement metrics to generate a **Composite Communication Score (0–10)** for each video.

---

## 🚀 Overview

Communication quality is subjective and cannot be measured directly. This project approximates communication effectiveness using quantitative linguistic and engagement indicators extracted from YouTube transcripts and metadata.

| Communication Trait | Programmatic Feature |
|---|---|
| Speaking clarity | Speech Rate (WPM) |
| Conciseness | Filler Word Ratio |
| Structured explanation | Instructional Keyword Density |
| Audience interaction | Engagement Score |
| Information density | Subtitle Coverage Ratio |
| Vocabulary richness | Lexical Diversity |
| Explanation complexity | Average Sentence Length |

---

## 🧩 Processing Pipeline

```
YouTube URL
    ↓
extract_video_id()       → Parse video ID from URL
    ↓
fetch_transcript()       → Retrieve subtitles via API
    ↓
fetch_metadata()         → Collect video statistics
    ↓
regex tokenization       → Clean and tokenize text
    ↓
metric functions         → Compute communication features
    ↓
run_pipeline()           → Export structured CSV dataset
```

---

## 📐 Features Extracted

| Feature | Formula | Why It Matters |
|---|---|---|
| **Speech Rate (WPM)** | Total Words / (Speech Duration × 60) | Measures speaking pace for clarity |
| **Filler Word Ratio** | Filler Words / Total Words × 100 | High filler usage reduces clarity |
| **Instructional Keyword Density** | Instructional Keywords / Total Words × 100 | Captures structured step-by-step explanations |
| **Audience Engagement Score** | 0.40×CTA + 0.35×SecondPerson + 0.25×Questions | Measures active audience interaction |
| **Subtitle Coverage Ratio** | Speech Duration / Video Duration × 100 | Indicates spoken content density |
| **Lexical Diversity (TTR)** | Unique Words / Total Words | Measures vocabulary richness |
| **Average Sentence Length** | Total Words / Number of Sentences | Approximates explanation complexity |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `Python` | Core implementation |
| `youtube-transcript-api` | Transcript extraction |
| `yt-dlp` | Metadata collection |
| `Pandas` | Data processing |
| `NumPy` | Numerical computation |
| `Regex` | Text preprocessing and tokenization |

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/a-communication-analytics.git
cd a-communication-analytics
pip install -r requirements.txt
```

---

## 📦 Usage

```python
from pipeline import run_pipeline

# Pass a list of YouTube video URLs
urls = [
    "https://www.youtube.com/watch?v=example1",
    "https://www.youtube.com/watch?v=example2"
]

run_pipeline(urls, output_path="output.csv")
```

The output CSV will contain one row per video with all extracted features and a composite communication score.

---

## ✅ Feature Validation

The extracted features can be validated using the following approaches:

- Collect human-labeled communication scores (1–10) for a sample of tutorial videos
- Compute Pearson or Spearman correlations between features and human ratings
- Train a regression model to predict communication scores and analyze feature importance
- Perform A/B evaluation by comparing predicted high-scoring vs low-scoring videos against real engagement metrics (likes, comments, watch time)

---

## 📁 Project Structure

```
a-communication-analytics/
│
├── pipeline.py            # Main pipeline runner
├── extractor.py           # Feature extraction functions
├── transcript.py          # Transcript and metadata fetchers
├── utils.py               # Preprocessing and tokenization helpers
├── requirements.txt       # Dependencies
├── output/
│   └── results.csv        # Sample output dataset
└── README.md
```

---

## 👩‍💻 Author

**Harshika Mhapsekar**
[LinkedIn]([https://linkedin.com/in/harshika-mhapsekar](https://www.linkedin.com/in/harshikamhapsekar/)) • [GitHub](https://github.com/your-username) • harshika.mhapsekar@gmail.com
