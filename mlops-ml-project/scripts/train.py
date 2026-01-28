import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import json
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.data import load_data
from src.model import build_model
from src.features import get_preprocessing_pipeline

def main():
    # Load config
    config_path = "config/train.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(config)

    # Preprocessing and Model Training
    print("Preprocessing and Training model...")
    pipeline = get_preprocessing_pipeline()
    model = build_model(config)
    
    # Create a full pipeline that includes preprocessing and the model
    from sklearn.pipeline import Pipeline as SkPipeline
    full_pipeline = SkPipeline(steps=pipeline.steps + [('model', model)])
    
    full_pipeline.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = full_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")

    # Artifacts path
    artifacts_path = config.get("paths", {}).get("artifacts_path", "artifacts/")
    os.makedirs(artifacts_path, exist_ok=True)

    # Save the FULL PIPELINE as the model
    model_path = config.get("paths", {}).get("model_path", os.path.join(artifacts_path, "model.joblib"))
    joblib.dump(full_pipeline, model_path)
    print(f"Full pipeline saved to {model_path}")

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro
    }
    metrics_path = config.get("paths", {}).get("metrics_path", os.path.join(artifacts_path, "metrics.json"))
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    cm_path = os.path.join(artifacts_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
