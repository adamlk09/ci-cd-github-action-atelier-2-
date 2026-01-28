from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

class ClippingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_limit_ = None
        self.upper_limit_ = None

    def fit(self, X, y=None):
        X = np.array(X)
        self.lower_limit_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_limit_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        X = np.array(X).copy()
        for i in range(X.shape[1]):
            X[:, i] = np.clip(X[:, i], self.lower_limit_[i], self.upper_limit_[i])
        return X

def get_preprocessing_pipeline():
    """
    Creates a preprocessing pipeline with median imputation, clipping, and standard scaling.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', ClippingTransformer()),
        ('scaler', StandardScaler())
    ])
    return pipeline
