import os
import pandas as pd
from catboost import CatBoostClassifier
from preprocessing import load_train_data, run_preproc

MODEL_DIR = "./models"
MODEL_PATH = f"{MODEL_DIR}/my_catboost.cbm"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading raw training data")
train_raw = pd.read_csv("./train_data/train.csv")

print("Running preprocessing")
train_processed = run_preproc(train_raw, train_raw)

X = train_processed
y = train_raw["target"]

print("Training CatBoost model")
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    loss_function="Logloss",
    verbose=100
)

model.fit(X, y)

model.save_model(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
