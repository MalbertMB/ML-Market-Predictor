"""
Project Name: ML-Predictor
File Name: base_model.py
Description: 
    This module defines the BaseStockModel abstract class, which serves as a blueprint for all stock prediction models in the ML-Predictor project.
    It ensures that all models implement the necessary methods for training and prediction, providing a consistent interface for model development and evaluation.
    
Author: Albert Marín Blasco
"""


from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseStockModel(ABC):
    """
    Abstract base class for all stock prediction models.
    Ensures a consistent interface across different algorithms.
    """
    
    def __init__(self, **kwargs):
        self.model = None
        self.params = kwargs

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Trains the model on the provided dataset."""
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generates predictions for the provided dataset."""
        pass