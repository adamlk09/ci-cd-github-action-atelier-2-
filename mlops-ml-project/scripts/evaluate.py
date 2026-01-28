import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import json
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from src.data import load_data
from src.model import load_model

def main():
    # Load config
    config_path = "config/train.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load data (only test set needed)
    print("Loading data...")
    _, X_test, _, y_test = load_data(config)

    # Load model
    model_path = config.get("paths", {}).get("model_path", "artifacts/model.joblib")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save report
    artifacts_path = config.get("paths", {}).get("artifacts_path", "artifacts/")
    os.makedirs(artifacts_path, exist_ok=True)
    
    report_path = config.get("paths", {}).get("report_path", os.path.join(artifacts_path, "report.json"))
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Evaluation complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
