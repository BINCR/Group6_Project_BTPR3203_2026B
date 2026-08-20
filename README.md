# BTPR3203 Python for Data Science - Semester B, 2026
## Group 6 Project: YouTube Comments Sentiment and Engagement Analysis

### Project Overview
This project investigates user engagement patterns and sentiment distribution within YouTube comments using Python data science libraries. By analyzing comment text length, interaction types (top-level comments vs. replies), and emotional polarity through machine learning, this study aims to uncover what drives audience interaction on video content.

---

### Group Members (Group 6)
* **University:** Southern University College, Johor, Malaysia
* **Course:** BTPR3203 Python for Data Science (Semester B, 2026)

---

### Repository Structure
```text
group6-youtube-analysis/
├── figure/                             # Directory containing all generated visualizations (Figures 1-6 & Graphs 3.1-3.3)
├── youtube_comments_with_sentiment.csv # Unified primary dataset with sentiment labels
├── youtube_comments_raw.csv            # Raw dataset collected via scraper
├── rq1_summary.csv                     # Statistical summary output for RQ1
├── rq2_summary.csv                     # Statistical summary output for RQ2
├── rq3_final_ml_predictions.csv        # Machine learning prediction output for RQ3
├── youtube_scrape.py                   # Python script for scraping YouTube comments
├── data_cleaning.py                    # Python script for data preprocessing and feature engineering
├── analysis.ipynb                      # Main Jupyter Notebook containing EDA, statistics, and machine learning
└── README.md                           # Project documentation