"""Data Download."""

import argparse
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import train_test_split

from logs import logger

parser = argparse.ArgumentParser()

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# parser.add_argument(
#     "--HOUSING_PATH",
#     help="Output path to save the dataset",
#     default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
# )
# args = parser.parse_args()
# HOUSING_PATH = args.HOUSING_PATH

HOUSING_PATH = 'D:/TAMLEP/assigns/1/mle-training/src/housing/data'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Download housing data."""
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logger.info(f"Downloaded and extracted data")


def load_housing_data(housing_path=HOUSING_PATH):
    """Read Housing data and return dataframe."""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def main():
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)

    housing = load_housing_data(HOUSING_PATH)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    train_set.to_csv(os.path.join(HOUSING_PATH, "train.csv"), index=False)
    test_set.to_csv(os.path.join(HOUSING_PATH, "valid.csv"), index=False)

    logger.info(f"Saved test and train datsets at {HOUSING_PATH}")

if __name__ == '__main__':
    main()