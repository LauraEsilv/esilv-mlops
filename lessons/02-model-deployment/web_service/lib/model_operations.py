# web_service/lib/model_operations.py
from typing import Any
from app_config import AppConfig

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from typing import List
from lib.models import Input
from lib.preprocesor import CATEGORICAL_COLS, encode_categorical_cols
from sklearn.base import BaseEstimator
import pandas as pd

def run_inference(data: List[Input], dv: DictVectorizer, model: BaseEstimator) -> np.ndarray:
    # Create a DataFrame with the necessary columns
    input_df =pd.DataFrame([x.dict() for x in data]) #pd.DataFrame([data.dict()])
    #input_df = compute_target(input_df)
    input_df = encode_categorical_cols(input_df)
    dict = input_df[CATEGORICAL_COLS].to_dict(orient="records")
    X = dv.transform(dict)
    result = model.predict(X)
    return result