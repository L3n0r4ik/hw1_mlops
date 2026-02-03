import pandas as pd
import logging
import os
from catboost import CatBoostClassifier
from preprocessing import preprocess

logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_model.cbm")


def load_model(model_path: str):
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
    return model


def make_pred(path_to_file: str, model_path: str) -> pd.DataFrame:
    logger.info("Reading input file: %s", path_to_file)
    raw_df = pd.read_csv(path_to_file, sep=",")
    processed_df = preprocess(raw_df)

    logger.info("Path to model: %s", MODEL_PATH)
    model = load_model(MODEL_PATH)

    preds = model.predict_proba(processed_df)[:, 1]

    submission = pd.DataFrame({
        "index": raw_df.index,
        "prediction": preds
    })

    logger.info("Prediction finished. Rows: %d", len(submission))
    return submission
