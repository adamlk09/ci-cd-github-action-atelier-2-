from sklearn.linear_model import LogisticRegression
import joblib
import os

def build_model(config):
    """
    Builds a Logistic Regression model based on configuration.
    """
    model_config = config.get("model", {})
    # Use LogisticRegression as requested in Part 3
    params = model_config.get("params", {})
    
    # Filtering params to only keep those valid for LogisticRegression 
    # (or just simple instantiation since we might have RF params in config)
    lr_params = {
        "random_state": params.get("random_state", 42),
        "max_iter": params.get("max_iter", 1000)
    }
    
    model = LogisticRegression(**lr_params)
    return model

def save_model(model, path):
    """
    Saves the model to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    """
    Loads the model from a file.
    """
    return joblib.load(path)
