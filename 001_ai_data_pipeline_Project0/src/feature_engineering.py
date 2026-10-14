import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
clean_path = os.path.join(BASE_DIR, "..", "data", "processed", "clean_posts.csv")
feature_path = os.path.join(BASE_DIR, "..", "data", "processed", "featured_posts.csv")


df = pd.read_csv(clean_path)


df["title_length"] = df["title"].apply(len)
df["body_length"] = df["body"].apply(len)
df["title_word_count"] = df["title"].apply(lambda x: len(x.split()))
df["body_word_count"] = df["body"].apply(lambda x: len(x.split()))


df.to_csv(feature_path, index=False)

print("Feature engineering completed successfully!")
