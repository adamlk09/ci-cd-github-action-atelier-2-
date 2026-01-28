import yaml
import json
import os
import sys
from sklearn.metrics import classification_report

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_data
from src.model import load_model

def evaluate():
    # Load config
    config_path = "config/train.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(config)
    
    # Load model
    model_path = config["paths"]["model_path"]
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run train.py first.")
        return
        
    model = load_model(model_path)
    
    # Predict
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save artifacts
    artifacts_path = config["paths"]["artifacts_path"]
    os.makedirs(artifacts_path, exist_ok=True)
    
    report_path = config["paths"]["report_path"]
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Evaluation complete. Report saved to {report_path}")

if __name__ == "__main__":
    evaluate()

if __name__ == "__main__":
    evaluate()
