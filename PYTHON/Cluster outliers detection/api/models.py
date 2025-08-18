from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoints.

    Attributes:
    - data: A list of feature vectors, where each inner list represents one sample.
    """
    data: List[List[float]]  # 2D list of floats representing input features
