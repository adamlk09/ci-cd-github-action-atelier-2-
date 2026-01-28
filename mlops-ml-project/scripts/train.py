import yaml
import os
import sys
import json
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_data
from src.features import get_preprocessing_pipeline
from src.model import build_model, save_model

def train():
    # Load config
    config_path = "config/train.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(config)
    
    # Preprocessing
    preprocessor = get_preprocessing_pipeline()
    
    # Model
    model = build_model(config)
    
    # Full Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Intermediate Evaluation for train.py artifacts
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    metrics = {
        "accuracy": acc,
        "f1_macro": f1
    }
    
    # Ensure artifacts directory exists
    artifacts_path = config["paths"]["artifacts_path"]
    os.makedirs(artifacts_path, exist_ok=True)
    
    # Save Model
    model_path = config["paths"]["model_path"]
    save_model(pipeline, model_path)
    
    # Save Metrics
    metrics_path = config["paths"]["metrics_path"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    cm_path = os.path.join(artifacts_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    print("Training complete.")

if __name__ == "__main__":
    train()
