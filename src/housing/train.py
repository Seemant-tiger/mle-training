"""Model training."""

import argparse
import os
import pickle

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin

from logs import logger

# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "DATA_PATH",
#     help="Path for input dataset",
#     default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
# )
# parser.add_argument(
#     "MODEL_PATH",
#     help="Path to save model files",
#     default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"),
# )
# args = parser.parse_args()
# DATA_PATH = args.DATA_PATH
# MODEL_PATH = args.MODEL_PATH

DATA_PATH = 'D:/TAMLEP/assigns/1/mle-training/src/housing/data'
MODEL_PATH = 'D:/TAMLEP/assigns/1/mle-training/src/housing/data'


def load_housing_dataset(data_path=DATA_PATH):
    """Read Housing data and return dataframe."""
    train_path = os.path.join(data_path, "train.csv")
    valid_path = os.path.join(data_path, "valid.csv")
    logger.info(f"Reading train and test data")
    return pd.read_csv(train_path), pd.read_csv(valid_path)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self): # no *args or **kargs
        pass
        
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    def transform(self, X, rooms_ix, bedrooms_ix, population_ix, households_ix):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

def main():
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

    col_names = "total_rooms", "total_bedrooms", "population", "households"
    rooms_ix, bedrooms_ix, population_ix, households_ix = [
        housing.columns.get_loc(c) for c in col_names] # get the column indices

    attr_adder = CombinedAttributesAdder()
    housing_extra_attribs = attr_adder.transform(housing.values, rooms_ix, bedrooms_ix, population_ix, households_ix)

    housing_extra_attribs = pd.DataFrame(
        housing_extra_attribs,
        columns=list(housing.columns)+["rooms_per_household", "population_per_household", "bedrooms_per_room"],
        index=housing.index)

    housing_tr = housing_extra_attribs.drop("ocean_proximity", axis=1)

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

    col_names = "total_rooms", "total_bedrooms", "population", "households"
    rooms_ix, bedrooms_ix, population_ix, households_ix = [
        X_test.columns.get_loc(c) for c in col_names] # get the column indices

    attr_adder = CombinedAttributesAdder()
    housing_extra_attribs = attr_adder.transform(X_test.values, rooms_ix, bedrooms_ix, population_ix, households_ix)

    housing_extra_attribs = pd.DataFrame(
        housing_extra_attribs,
        columns=list(X_test.columns)+["rooms_per_household", "population_per_household", "bedrooms_per_room"],
        index=X_test.index)

    X_test = housing_extra_attribs

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

if __name__ == '__main__':
    main()