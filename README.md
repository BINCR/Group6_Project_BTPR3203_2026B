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
├── youtube_comments_with_sentiment.csv # Final unified dataset with sentiment labels (input to analysis.ipynb)
├── rq1_summary.csv                     # Statistical summary output for RQ1
├── rq2_summary.csv                     # Statistical summary output for RQ2
├── rq3_final_ml_predictions.csv        # Machine learning prediction output for RQ3
├── youtube_scrape.py                   # Script to scrape raw YouTube comments
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

`requirements.txt` should contain at minimum:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
textblob
youtube-comment-downloader
jupyter
```

After installing, download the TextBlob corpora once:

```bash
python -m textblob.download_corpora
```

---

## How to Run

The pipeline runs in three stages, in order.

### Stage 1 — Scrape the raw data

```bash
python youtube_scrape.py
```

Collects raw comments from the target YouTube video and saves them as `youtube_comments_raw.csv` (fields: `comment_id`, `comment`, `author`, `like_count`, `time_text`, `reply`).

### Stage 2 — Clean, engineer features, and label sentiment

```bash
python data_cleaning.py
```

This script:
- loads `youtube_comments_raw.csv`,
- removes hyperlinks, line breaks, and duplicate whitespace from comment text (`clean_comment`),
- converts the raw `like_count` string (e.g. `"2.7K"`) into a numeric field (`like_count_numeric`),
- derives `word_count`, `character_count`, and `comment_type` (Top-level / Reply),
- drops empty comments and duplicate `comment_id` rows,
- assigns a `sentiment` label (Positive / Negative / Neutral) to each comment using TextBlob's lexicon-based polarity score,
- saves the final dataset as `youtube_comments_with_sentiment.csv`.

> **Note:** the version of `data_cleaning.py` currently committed to this repository performs cleaning and feature engineering but does **not yet include the TextBlob sentiment-labelling step**, and it writes its output to `youtube_comments_cleaned.csv` rather than `youtube_comments_with_sentiment.csv`. The `youtube_comments_with_sentiment.csv` file included in this repo was produced with an additional sentiment-labelling step that still needs to be merged into `data_cleaning.py` so the full pipeline can be reproduced end-to-end from a single script. This is a known limitation — see "Known Issues" below.

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

> **Note:** the first code cell of the notebook must define the output folder before the RQ1 cell runs, e.g.:
> ```python
> figure_dir = Path("Figure")
> figure_dir.mkdir(exist_ok=True)
> ```
> The currently committed notebook only defines `charts_dir` in this cell, not `figure_dir`, which the RQ1/RQ2/RQ3 cells rely on to save images. Running "Restart & Run All" on the current version will raise `NameError: name 'figure_dir' is not defined` on the RQ1 cell. Add the two lines above to the first code cell before submission/demo to make the notebook fully reproducible from a clean environment.

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

- **Data cleaning:** removal of hyperlinks, line breaks, and duplicate whitespace; conversion of like counts to numeric type; removal of empty/duplicate comments.
- **Feature engineering:** `word_count`, `character_count`, `comment_type` (top-level vs. reply), and `sentiment` (TextBlob lexicon-based labelling).
- **Statistical analysis:** Pearson/Spearman correlation, IQR-based outlier removal, Welch's independent t-test.
- **Machine learning:** TF-IDF vectorisation (1,000 features, English stop words removed) → Logistic Regression classifier, evaluated against a Multinomial Naive Bayes baseline (Accuracy: 63.81% vs. 60.95%), using an 80/20 stratified train-test split (`random_state=42`).

Full methodology, findings, and interpretation are available in the project report.

---

### Known Issues

These are documented transparently for grading/reproducibility purposes and are being addressed before final submission:

1. `data_cleaning.py` does not yet include the TextBlob sentiment-labelling step and writes to a differently-named output file than the one consumed by `analysis.ipynb`. The two need to be reconciled into a single reproducible script.
2. `analysis.ipynb` requires a `figure_dir` variable to be defined in the first code cell (see "How to Run", Stage 3) — without it, "Restart & Run All" fails on the RQ1 cell.
3. `requirements.txt` should be generated with `pip freeze > requirements.txt` once the environment used to produce the final results is finalised, to guarantee identical package versions.

---

### References

1. BBC News. (2023). *How 'super' El Niño could bring chaos to the world's weather* [Video]. YouTube. https://youtube.com/watch?v=UEseLvpl9ss
2. McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference* (pp. 51–56).
3. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
