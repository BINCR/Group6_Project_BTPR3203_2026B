import pandas as pd
import re
from pathlib import Path

# 1. LOAD DATA
def load_data(file_path):

    df = pd.read_csv(file_path)

    print("=" * 60)
    print("RAW DATASET")
    print("=" * 60)
    print(f"Records : {len(df)}")
    print(f"Columns : {len(df.columns)}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate comment IDs:")
    print(df["comment_id"].duplicated().sum())

    return df

# 2. CONVERT LIKE COUNT
def convert_like_count(value):

    """
    Convert YouTube like-count text into numeric values.

    Examples:
    2.7K -> 2700
    1.8K -> 1800
    566  -> 566
    """

    if pd.isna(value):
        return 0

    value = str(value).strip().upper()

    try:
        if value.endswith("K"):
            return int(float(value[:-1]) * 1000)
        elif value.endswith("M"):
            return int(float(value[:-1]) * 1000000)
        else:
            return int(float(value))
    except ValueError:
        return 0

# 3. CLEAN COMMENT TEXT
def clean_comment(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove line breaks
    text = text.replace("\n", " ")

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# 4. FEATURE ENGINEERING

def create_features(df):

# Clean comment
    df["clean_comment"] = df["comment"].apply(clean_comment)

# Numeric like count
    df["like_count_numeric"] = (df["like_count"].apply(convert_like_count))

# Word count
    df["word_count"] = (df["clean_comment"].str.split().str.len())

# Character count
    df["character_count"] = (df["clean_comment"].str.len())

# Comment type
    df["comment_type"] = df["reply"].apply(lambda x: "Reply" if bool(x) else "Top-level")

    return df

# 5. REMOVE INVALID RECORDS
def clean_records(df):

    original_count = len(df)

    # Remove missing/empty comments
    df = df[df["clean_comment"].str.strip() != ""].copy()

    # Remove duplicate comment IDs
    df = df.drop_duplicates(subset="comment_id",keep="first")

    removed = original_count - len(df)

    print("\n" + "=" * 60)
    print("CLEANING RESULTS")
    print("=" * 60)
    print(f"Original records : {original_count}")
    print(f"Removed records  : {removed}")
    print(f"Final records    : {len(df)}")

    return df

# 6. DATA QUALITY SUMMARY
def show_summary(df):

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print("\nComment types:")
    print(df["comment_type"].value_counts())
    print("\nWord count statistics:")
    print(df["word_count"].describe())
    print("\nLike count statistics:")
    print(df["like_count_numeric"].describe())

    print("\nTop 10 most-liked comments:")

    columns = ["comment","like_count_numeric","word_count","comment_type"]

    print(df.nlargest(10,"like_count_numeric")[columns].to_string(index=False))

# 7. SAVE CLEANED DATA
def save_data(df, output_path):

    df.to_csv(output_path,index=False,encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"Cleaned dataset saved to:")
    print(output_path)

# MAIN
def main():

    # Get the folder where data_cleaning.py is located
    base_dir = Path(__file__).resolve().parent

    input_file = base_dir / "youtube_comments_raw.csv"
    output_file = base_dir / "youtube_comments_cleaned.csv"

    print("Python file location:")
    print(base_dir)
    print("\nLooking for input file:")
    print(input_file)
    print("\nFile exists:")
    print(input_file.exists())

    df = load_data(input_file)
    df = create_features(df)
    df = clean_records(df)
    show_summary(df)
    save_data(df, output_file)

if __name__ == "__main__":
    main()
