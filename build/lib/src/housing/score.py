"""Model Scoring."""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .logs import logger

parser = argparse.ArgumentParser()

parser.add_argument(
    "DATA_PATH",
    help="Path for input dataset",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
parser.add_argument(
    "MODEL_PATH",
    help="Path to save model files",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"),
)
args = parser.parse_args()
DATA_PATH = args.DATA_PATH
MODEL_PATH = args.MODEL_PATH


def load_housing_data_processed(data_path=DATA_PATH):
    """Read Housing data and return dataframe."""
    train_path = os.path.join(data_path, "train_processed.csv")
    valid_path = os.path.join(data_path, "valid_processed.csv")
    logger.info(f"Reading processed data for results validation")
    return pd.read_csv(train_path), pd.read_csv(valid_path)


train_set, test_set = load_housing_data_processed(DATA_PATH)

with open(os.path.join(MODEL_PATH, "linear_model.pkl"), "rb") as file:
    lin_reg = pickle.load(file)

X_train = train_set.drop("median_house_value", axis=1)
y_train = train_set["median_house_value"].copy()

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()


housing_predictions = lin_reg.predict(X_train)
lin_mse = mean_squared_error(y_train, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

lin_mae = mean_absolute_error(y_train, housing_predictions)
lin_mae

final_predictions = lin_reg.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_mae = mean_absolute_error(y_test, final_predictions)
final_mae

logger.info(f"Done.")
