import logging
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(BASE_DIR, "..", "logs")
os.makedirs(log_path, exist_ok=True)

LOG_FILE = os.path.join(log_path, "pipeline.log")


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)