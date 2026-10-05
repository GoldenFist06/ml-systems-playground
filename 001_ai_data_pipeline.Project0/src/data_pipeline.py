import requests
import csv
import os
from logging_config import logging  


url = "https://jsonplaceholder.typicode.com/posts"
logging.info("Sending request to API...")


response = requests.get(url)
logging.info("API response received.")


data = response.json()
logging.info(f"Fetched {len(data)} records from API.")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "..", "data", "raw", "posts_data.csv")

logging.info(f"Saving data to CSV at: {csv_path}")

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    

    writer.writerow(["userId", "id", "title", "body"])
    

    for item in data:
        writer.writerow([item["userId"], item["id"], item["title"], item["body"]])

logging.info("Data saved successfully!")
print("Data saved successfully!")
