from pydantic import BaseModel

# Define a data model for the request body
# We're using StrictStr to ensure that the name is a string
# More information here: https://stackoverflow.com/questions/72263682/checking-input-data-types-in-pydantic
class Input(BaseModel):
    PULocationID: int
    DOLocationID: int
    passenger_count: int


class Prediction(BaseModel):
    result: int



