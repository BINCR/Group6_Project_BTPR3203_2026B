# BTPR3203 Python for Data Science - Semester B, 2026

## Group 6 Project: YouTube Comments Sentiment and Engagement Analysis

### Project Overview

This project investigates user engagement patterns and sentiment distribution within YouTube comments using Python data science libraries. By analyzing comment text length, interaction types (top-level comments vs. replies), and emotional polarity through machine learning, this study aims to uncover what drives audience interaction on video content.

The dataset was collected from the public comment section of the BBC News YouTube video *"How 'super' El Niño could bring chaos to the world's weather"*, in relation to the group's chosen problem domain of climate-related public discourse.

---

### Group Members (Group 6)

- Mah Chee Peng (B230224C)
- Tan Chun Wai (B230289C)
- Chew Rong Bin (B240074A)

- **University:** Southern University College, Johor, Malaysia
- **Course:** BTPR3203 Python for Data Science (Semester B, 2026)

---

### Repository Structure

```
Group6_Project_BTPR3203_2026B/
├── Figure/                             # All generated visualizations (Figures 1-6 & Graphs 3.1-3.3)
├── youtube_comments_with_sentiment.csv # Unified primary dataset with sentiment labels (post-processing)
├── rq1_summary.csv                     # Statistical summary output for RQ1
├── rq2_summary.csv                     # Statistical summary output for RQ2
├── rq3_final_ml_predictions.csv        # Machine learning prediction output for RQ3
├── youtube_scrape.py                   # Scrapes raw comments from the target YouTube video
├── data_cleaning.py                    # Cleans raw data and performs feature engineering
├── analysis.ipynb                      # Main notebook: EDA, statistical tests, and ML classification
├── requirements.txt                    # Python package dependencies
└── README.md                           # Project documentation
```

---

### Requirements

- Python 3.10 or later
- Recommended: a virtual environment (`venv` or `conda`)

Install dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` should contain:

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
textblob
youtube-comment-downloader
jupyter
```

---

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/BINCR/Group6_Project_BTPR3203_2026B.git
   cd Group6_Project_BTPR3203_2026B
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### How to Run

The pipeline must be run in the following order, since each script depends on the output of the previous one:

1. **Scrape raw comments** (produces `youtube_comments_raw.csv`):
   ```bash
   python youtube_scrape.py
   ```
2. **Clean data and engineer features** (produces `youtube_comments_cleaned.csv`):
   ```bash
   python data_cleaning.py
   ```
3. **Run the full analysis** (EDA, statistical tests, sentiment labeling, and ML classification):
   Open `analysis.ipynb` in Jupyter Notebook or JupyterLab and run all cells in order:
   ```bash
   jupyter notebook analysis.ipynb
   ```
   This notebook produces:
   - `youtube_comments_with_sentiment.csv`
   - `rq1_summary.csv`
   - `rq2_summary.csv`
   - `rq3_final_ml_predictions.csv`
   - All figures/graphs saved to the `Figure/` directory

> Note: Since `youtube_comments_raw.csv` and `youtube_comments_cleaned.csv` are intermediate files, they are not committed to this repository. Running Steps 1–3 in order will regenerate them locally.

---

### Output Files

| File | Description |
|---|---|
| `youtube_comments_with_sentiment.csv` | Final cleaned dataset with all engineered features and sentiment labels |
| `rq1_summary.csv` | Descriptive statistics and correlation results for RQ1 (word count vs. likes) |
| `rq2_summary.csv` | Descriptive statistics and t-test results for RQ2 (top-level comments vs. replies) |
| `rq3_final_ml_predictions.csv` | Model predictions on the held-out test set for RQ3 (sentiment classification) |
