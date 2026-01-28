from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def get_preprocessing_pipeline():
    """
    Creates a preprocessing pipeline with median imputation and standard scaling.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return pipeline
