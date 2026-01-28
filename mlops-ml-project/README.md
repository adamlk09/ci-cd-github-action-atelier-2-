mlops-ml-project/
├── .github/workflows/  # CI/CD pipelines
│   ├── ci.yml          # Continuous Integration
│   └── train.yml       # Continuous Training
├── config/             # Configuration files
│   └── train.yaml      # Training parameters
├── data/               # Dataset
│   └── dataset.csv     # Iris dataset
├── src/                # Source code
│   ├── data.py         # Data loading
│   ├── features.py     # Preprocessing
│   ├── model.py        # Model training
│   └── evaluate.py     # Model evaluation
├── scripts/            # Execution scripts
│   ├── train.py        # Train pipeline
│   └── evaluate.py     # Evaluate pipeline
├── Dockerfile          # Docker configuration
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized execution)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd mlops-ml-project
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage

### Train the Model

```bash
python scripts/train.py
```

This will:
1. Load the Iris dataset
2. Preprocess the data
3. Train a Logistic Regression model
4. Save the model to `artifacts/model.joblib`

### Evaluate the Model

```bash
python scripts/evaluate.py
```

This will:
1. Load the trained model
2. Evaluate on the test set
3. Save metrics to `artifacts/metrics.json` (Accuracy and F1-macro)
4. Save classification report to `artifacts/report.json`
5. Generate `artifacts/confusion_matrix.png` (from `train.py`)

### Using Docker

Build the Docker image:

```bash
docker build -t mlops-iris .
```

Run the training pipeline:

```bash
docker run -v $(pwd):/app mlops-iris python scripts/train.py
```

## ⚙️ Configuration

Edit `config/train.yaml` to customize:

```yaml
data:
  test_size: 0.2
  random_state: 42

model:
  type: "LogisticRegression"
  params:
    random_state: 42
    max_iter: 1000

paths:
  data_path: "data/dataset.csv"
  artifacts_path: "artifacts/"
  model_path: "artifacts/model.joblib"
  metrics_path: "artifacts/metrics.json"
  report_path: "artifacts/report.json"
```

## 🔄 CI/CD Automation

### Continuous Integration (CI)

Triggers on:
- Push to `main` branch
- Pull requests to `main` branch

Runs:
- Linting (flake8)
- Unit tests (pytest)
- Dependency checks

### Continuous Training (CT)

Triggers on:
- Push to `main` branch

Runs:
- Full training pipeline
- Model evaluation
- Artifact storage

## 🧪 Testing

Run unit tests:

```bash
pytest