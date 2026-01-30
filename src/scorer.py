import pandas as pd
import logging
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

model = CatBoostClassifier()
model.load_model('./models/my_catboost.cbm')

logger.info('Pretrained model imported successfully...')

def make_pred(dt, path_to_file):
    preds = model.predict_proba(processed_df)[:, 1]

    submission = pd.DataFrame({
        "index": raw_df.index,
        "prediction": preds
    })

    logger.info('Prediction complete for file: %s', path_to_file)

    return submission