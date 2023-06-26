from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from src.jiyeretal import SecondaryFeatureSet


def train(X: np.ndarray, y: list[float]):
    return LinearRegression().fit(X, y)
