import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd

from src.housing.ingest_data import load_housing_data

def test_returns_dataframe():
    df = load_housing_data()
    assert isinstance(df, pd.DataFrame)