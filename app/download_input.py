import requests
import os
import logging

logger = logging.getLogger(__name__)

GDRIVE_FILE_ID = "1bAKqG06t5hg0_W-v7wChERNOH9H5Mq0j"
OUTPUT_PATH = "/app/input/test.csv"

def download_file_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    logger.info("Downloading test.csv from Google Drive...")
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(response.content)
        logger.info("test.csv downloaded successfully to %s", dest_path)
    else:
        logger.error("Failed to download file: %s, status code: %s", url, response.status_code)
        raise RuntimeError("Failed to download test.csv")

if __name__ == "__main__":
    download_file_from_gdrive(GDRIVE_FILE_ID, OUTPUT_PATH)
