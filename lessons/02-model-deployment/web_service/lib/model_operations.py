# web_service/lib/model_operations.py
from typing import Any
from app_config import AppConfig
import pickle

def load_pickle(path: str):
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from typing import List
from lib.models import Input

def extract_x_y(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    dv: DictVectorizer = None,
    with_target: bool = True,
) -> dict:

    if categorical_cols is None:
        categorical_cols = ["PULocationID", "DOLocationID", "passenger_count"]
    dicts = df[categorical_cols].to_dict(orient="records")

    y = None
    if with_target:
        if dv is None:
            dv = DictVectorizer()
            dv.fit(dicts)
        y = df["duration"].values

    x = dv.transform(dicts)
    return x, y, dv

CATEGORICAL_COLS = ["PULocationID", "DOLocationID", "passenger_count"]

#def compute_target(
#    df: pd.DataFrame,
#    pickup_column: int = "PULocationID",
#    dropoff_column: int = "DOLocationID",
#) -> pd.DataFrame:
#    df["duration"] = df[dropoff_column] - df[pickup_column] /60
    #df["duration"] = df["duration"].dt.total_seconds() / 60
#    return df

#def filter_outliers(df: pd.DataFrame, min_duration: int = 1, max_duration: int = 60) -> pd.DataFrame:
#    return df[df["duration"].between(min_duration, max_duration)]

def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = ["PULocationID", "DOLocationID", "passenger_count"]
    df[categorical_cols] = df[categorical_cols].fillna(-1).astype("int")
    df[categorical_cols] = df[categorical_cols].astype("str")
    return df


def run_inference(data: Input):
    # Create a DataFrame with the necessary columns
    model = load_pickle(AppConfig.PATH_TO_PIPELINE)
    input_df = pd.DataFrame([data.dict()])
    #input_df = compute_target(input_df)
    input_df = encode_categorical_cols(input_df)
    dict = input_df[CATEGORICAL_COLS].to_dict(orient="records")
    X = input_df.transform(dict)
    extract_x_y(input_df)
    result = model.predict(X)
    return result