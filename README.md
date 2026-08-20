# BTPR3203 Python for Data Science — Semester B, 2026

## Group 6 Project: YouTube Comments Sentiment and Engagement Analysis

### Project Overview

This project investigates user engagement patterns and sentiment distribution within YouTube comments, using the comment section of a BBC News video on the resurgence of El Niño as the case study. By analysing comment length, interaction type (top-level comment vs. reply), and emotional polarity through machine learning, this study explores what drives audience engagement and how accurately automated sentiment classification can capture public reaction to climate-related news.

The project addresses three research questions:

- **RQ1:** Is there a correlation between the word count of a comment and the number of likes it receives?
- **RQ2:** Do top-level comments receive more likes than replies?
- **RQ3:** How accurately can machine learning classify YouTube comments into Positive, Negative, and Neutral sentiment categories?

---

### Group Members (Group 6)

| Name | Student ID |
|---|---|
| Mah Chee Peng | B230224C |
| Tan Chun Wai | B230289C |
| Chew Rong Bin | B240074A |

- **University:** Southern University College, Johor, Malaysia
- **Course:** BTPR3203 Python for Data Science (Semester B, 2026)
- **Lecturer:** Nur Shamilla Binti Selamat

---

### Repository Structure

```
Group6_Project_BTPR3203_2026B/
├── Figure/                             # All generated visualisations (Figures 1-6 & Graphs 3.1-3.3)
├── youtube_comments_with_sentiment.csv # Unified primary dataset with sentiment labels
├── rq1_summary.csv                     # Statistical summary output for RQ1
├── rq2_summary.csv                     # Statistical summary output for RQ2
├── rq3_final_ml_predictions.csv        # Machine learning prediction output for RQ3
├── youtube_scrape.py                   # Script to scrape YouTube comments
├── data_cleaning.py                    # Script for data preprocessing and feature engineering
├── analysis.ipynb                      # Main notebook: EDA, statistics, and machine learning
├── requirements.txt                    # Python package dependencies
└── README.md                           # Project documentation
```

---

### Dataset

- **Source:** Public comments scraped from the BBC News YouTube video *"How 'super' El Niño could bring chaos to the world's weather"*.
- **Size:** 1,572 records, 12 fields (raw metadata, cleaned attributes, and engineered variables).
- **Key fields:** `comment_id`, `comment`, `clean_comment`, `author`, `like_count_numeric`, `word_count`, `character_count`, `reply`, `comment_type`, `sentiment`.
- Full field descriptions and known limitations are documented in Section 4 of the project report.

---

## Setup Instructions

### 1. Prerequisites

- Python 3.9 or later
- pip (or conda)

### 2. Clone the repository

```bash
git clone https://github.com/BINCR/Group6_Project_BTPR3203_2026B.git
cd Group6_Project_BTPR3203_2026B
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet present in the repository, install the core packages directly:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn textblob jupyter
python -m textblob.download_corpora
```

> Add/adjust this list to match whichever scraping library `youtube_scrape.py` actually imports (e.g. `google-api-python-client`, `selenium`, or `youtube-comment-downloader`), and pin exact versions with `pip freeze > requirements.txt` once the environment is finalised.

---

## How to Run

The pipeline is designed to be run in three stages, in order:

### Stage 1 — Scrape the data

```bash
python youtube_scrape.py
```

This collects raw comments from the target YouTube video and saves them as a CSV file.

### Stage 2 — Clean and engineer features

```bash
python data_cleaning.py
```

This handles missing values, converts data types, removes noise (links, line breaks, duplicate spaces) from comment text, and generates the engineered fields (`word_count`, `character_count`, `comment_type`, `sentiment`). The output is `youtube_comments_with_sentiment.csv`.

### Stage 3 — Run the analysis

```bash
jupyter notebook analysis.ipynb
```

Run all cells in order (**Kernel → Restart & Run All**). The notebook performs:

- Exploratory data analysis and descriptive statistics
- Correlation analysis (Pearson, Spearman) and outlier detection (IQR) for RQ1
- Group comparison and Welch's t-test for RQ2
- TF-IDF vectorisation and Logistic Regression sentiment classification for RQ3
- Generation of all visualisations (saved to `Figure/`) and summary outputs (`rq1_summary.csv`, `rq2_summary.csv`, `rq3_final_ml_predictions.csv`)

---

### Outputs

| File | Description |
|---|---|
| `youtube_comments_with_sentiment.csv` | Cleaned dataset with all engineered features and sentiment labels |
| `rq1_summary.csv` | Descriptive statistics and correlation results for word count vs. likes |
| `rq2_summary.csv` | Comparison statistics for top-level comments vs. replies |
| `rq3_final_ml_predictions.csv` | Test-set predictions from the Logistic Regression sentiment classifier |
| `Figure/` | All charts referenced in the project report (Figures 1–6, Graphs 3.1–3.3) |

---

### Methodology Summary

- **Data cleaning:** removal of hyperlinks, line breaks, and duplicate whitespace; conversion of like counts to numeric type.
- **Feature engineering:** `word_count`, `character_count`, `comment_type` (top-level vs. reply), and `sentiment` (TextBlob lexicon-based labelling).
- **Statistical analysis:** Pearson/Spearman correlation, IQR-based outlier removal, Welch's independent t-test.
- **Machine learning:** TF-IDF vectorisation (1,000 features, English stop words removed) → Logistic Regression classifier, evaluated against a Multinomial Naive Bayes baseline (Accuracy: 63.81% vs. 60.95%), using an 80/20 stratified train-test split (`random_state=42`).

Full methodology, findings, and interpretation are available in the project report.

---

### References

1. BBC News. (2023). *How 'super' El Niño could bring chaos to the world's weather* [Video]. YouTube. https://youtube.com/watch?v=UEseLvpl9ss
2. McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference* (pp. 51–56).
3. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
