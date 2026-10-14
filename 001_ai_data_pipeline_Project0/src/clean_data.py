import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(BASE_DIR, "..", "data", "raw", "posts_data.csv")
processed_path = os.path.join(BASE_DIR, "..", "data", "processed", "clean_posts.csv")


df = pd.read_csv(raw_path)


df = df.drop_duplicates()  
df.columns = df.columns.str.lower()   


df.to_csv(processed_path, index=False)

print("Clean data saved successfully!")
