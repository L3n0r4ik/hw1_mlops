import os
import sys
import pandas as pd
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import subprocess

sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data, run_preproc
from scorer import make_pred

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_file = "/app/input/test.csv"
        self.output_dir = "/app/output"
        self.train = load_train_data()
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path).drop(columns=['name_1', 'name_2', 'street', 'post_code'])

            logger.info('Starting preprocessing')
            processed_df = run_preproc(self.train, input_df)
            
            logger.info('Making prediction')
            submission = make_pred(processed_df, file_path)
            
            logger.info('Preparing submission file')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}.csv"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Predictions saved to: %s', output_filename)

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    
    import download_input
    download_input.download_file_from_gdrive(
        download_input.GDRIVE_FILE_ID, 
        download_input.OUTPUT_PATH
    )

    service = ProcessingService()
    service.process_single_file(service.input_file)
