"""Model training."""

import argparse
import os
import pickle

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

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


def load_housing_dataset(data_path=DATA_PATH):
    """Read Housing data and return dataframe."""
    train_path = os.path.join(data_path, "train.csv")
    valid_path = os.path.join(data_path, "valid.csv")
    logger.info(f"Reading train and test data")
    return pd.read_csv(train_path), pd.read_csv(valid_path)


train_set, test_set = load_housing_dataset(DATA_PATH)


housing = train_set.drop(
    "median_house_value", axis=1
)  # drop labels for training set
housing_labels = train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

housing_tr["rooms_per_household"] = (
    housing_tr["total_rooms"] / housing_tr["households"]
)

housing_tr["rooms_per_household"] = (
    housing_tr["total_rooms"] / housing_tr["households"]
)

housing_tr["bedrooms_per_room"] = (
    housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
)
housing_tr["population_per_household"] = (
    housing_tr["population"] / housing_tr["households"]
)

housing_cat = housing[["ocean_proximity"]]

housing_prepared = housing_tr.join(
    pd.get_dummies(housing_cat, drop_first=True)
)

housing_prepared_tr = housing_prepared.copy()
housing_prepared_tr["median_house_value"] = housing_labels

logger.info(f"Training Linear Model.")

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


housing_predictions = lin_reg.predict(housing_prepared)


X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(
    X_test_prepared, columns=X_test_num.columns, index=X_test.index
)
X_test_prepared["rooms_per_household"] = (
    X_test_prepared["total_rooms"] / X_test_prepared["households"]
)
X_test_prepared["bedrooms_per_room"] = (
    X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
)
X_test_prepared["population_per_household"] = (
    X_test_prepared["population"] / X_test_prepared["households"]
)

X_test_cat = X_test[["ocean_proximity"]]

X_test_prepared = X_test_prepared.join(
    pd.get_dummies(X_test_cat, drop_first=True)
)

X_test_prepared_tr = X_test_prepared.copy()
X_test_prepared_tr["median_house_value"] = y_test

housing_prepared_tr.to_csv(
    os.path.join(DATA_PATH, "train_processed.csv"), index=False
)
X_test_prepared_tr.to_csv(
    os.path.join(DATA_PATH, "valid_processed.csv"), index=False
)

with open(os.path.join(MODEL_PATH, "linear_model.pkl"), "wb") as file:
    pickle.dump(lin_reg, file)

logger.info(
    f"Saved processed datasets at {DATA_PATH} and model pickle at {MODEL_PATH}"
)
