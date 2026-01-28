import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

def load_data(config):
    """
    Loads data from Iris dataset or a CSV file based on config.
    """
    data_path = config.get("paths", {}).get("data_path")
    
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        print("Loading Iris dataset from sklearn")
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name="target")
    
    test_size = config.get("data", {}).get("test_size", 0.2)
    random_state = config.get("data", {}).get("random_state", 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
