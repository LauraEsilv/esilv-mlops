import pickle
from functools import lru_cache

@lru_cache
def load_preprocessor(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@lru_cache
def load_pickle(path: str):
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj