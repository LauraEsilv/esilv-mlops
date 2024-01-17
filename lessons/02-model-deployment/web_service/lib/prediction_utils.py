
import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from typing import List
from scipy.sparse import csr_matrix


# Train Model

def train_model(x_train: csr_matrix, y_train: np.ndarray):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    return lr


# Predict

def predict(path: str, model: LinearRegression):
    return model.predict(path)


#def predict_duration(input_data: csr_matrix, model: LinearRegression):
#    return model.predict(input_data)
  