from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd

# 1. VIDEO URL
VIDEO_URL = "https://www.youtube.com/watch?v=UEseLvpl9ss"

# 2. DOWNLOAD COMMENTS
downloader = YoutubeCommentDownloader()
comments = []

print("Starting comment download...")

for comment in downloader.get_comments_from_url(VIDEO_URL, sort_by=0):

    comments.append({
        "comment_id": comment.get("cid"),
        "comment": comment.get("text"),
        "author": comment.get("author"),
        "like_count": comment.get("votes"),
        "time_text": comment.get("time"),
        "reply": comment.get("reply"),
    })

    if len(comments) % 100 == 0:
        print(f"Collected: {len(comments)} comments")

# 3. CREATE DATAFRAME
df = pd.DataFrame(comments)

print("\nTotal comments:", len(df))
print(df.head())
print(df.columns)

# 4. SAVE RAW DATA
df.to_csv("youtube_comments_raw.csv", index=False, encoding="utf-8-sig")

print("\nSaved as youtube_comments_raw.csv")
