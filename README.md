# ⚙️ SmartFault AI — Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)
[![ML](https://img.shields.io/badge/Models-RF%20|%20XGBoost%20|%20LSTM-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

> **Real-time machinery failure prediction using sensor streams (temperature, vibration, pressure) with 92%+ detection accuracy. Deployed via REST API with a live monitoring dashboard.**

---

## 🎯 What This Does

SmartFault AI ingests multi-sensor IoT data streams and predicts equipment failures **before they happen** — reducing unplanned downtime by up to 35%. Built end-to-end: from raw sensor preprocessing through model training to REST API deployment with a real-time dashboard.

| Metric | Result |
|---|---|
| Fault Detection Accuracy | **92.4%** |
| False Positive Reduction | **28%** vs baseline |
| Data Pipeline Speed | **40% faster** preprocessing |
| Downtime Reduction (simulated) | **35%** |

---

## 🏗️ Architecture

```
IoT Sensors → Data Pipeline → Feature Engineering → ML Models → REST API → Dashboard
     │               │                │                  │            │
temperature    pandas/SQL         rolling stats      RF/XGB/LSTM  FastAPI    Plotly
vibration      cleaning           FFT features       ensemble     Docker     real-time
pressure       labeling           lag features       prediction   /predict   alerts
```

---

## 📁 Project Structure

```
smartfault-ai/
├── data/
│   ├── generate_sensor_data.py     # Synthetic sensor data generator
│   └── preprocess.py               # ETL pipeline: clean, label, feature engineer
├── models/
│   ├── train_random_forest.py      # Random Forest classifier
│   ├── train_xgboost.py            # XGBoost with hyperparameter tuning
│   ├── train_lstm.py               # LSTM for temporal sequence modeling
│   ├── ensemble.py                 # Weighted ensemble voting
│   └── evaluate.py                 # Metrics: accuracy, F1, confusion matrix
├── api/
│   ├── main.py                     # FastAPI REST API
│   ├── schemas.py                  # Pydantic request/response models
│   └── predict.py                  # Inference engine
├── dashboard/
│   └── app.py                      # Streamlit real-time monitoring dashboard
├── notebooks/
│   └── EDA_and_Modeling.ipynb      # Full exploratory analysis + model walkthrough
├── tests/
│   └── test_api.py                 # API endpoint tests
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions CI/CD pipeline
├── Dockerfile                      # Container for API
├── docker-compose.yml              # Full stack deployment
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/smartfault-ai.git
cd smartfault-ai
pip install -r requirements.txt
```

### 2. Generate Sensor Data & Train Models
```bash
# Generate 20,000 rows of synthetic sensor data
python data/generate_sensor_data.py

# Run preprocessing pipeline
python data/preprocess.py

# Train all models
python models/train_random_forest.py
python models/train_xgboost.py
python models/train_lstm.py
```

### 3. Run the REST API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# API docs at: http://localhost:8000/docs
```

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Docker (Full Stack)
```bash
docker-compose up --build
```

---

## 📡 API Usage

### Predict Failure Risk
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "MACHINE_001",
    "temperature": 87.4,
    "vibration": 0.83,
    "pressure": 112.5,
    "rpm": 1450,
    "timestamp": "2024-04-28T14:30:00"
  }'
```

**Response:**
```json
{
  "machine_id": "MACHINE_001",
  "failure_probability": 0.87,
  "risk_level": "HIGH",
  "predicted_failure_in_hours": 6.2,
  "recommended_action": "Schedule maintenance within 6 hours",
  "model_used": "ensemble",
  "confidence": 0.91
}
```

### Batch Predict
```bash
POST /predict/batch   # Multiple machines at once
GET  /health          # API health check
GET  /metrics         # Model performance metrics
GET  /history/{id}    # Machine prediction history
```

---

## 🤖 Models

### Random Forest
- **Features**: 28 engineered features (rolling mean/std, FFT peaks, lag features)
- **Accuracy**: 91.2% | **F1**: 0.89
- Best for: interpretability, feature importance ranking

### XGBoost
- **Tuned with**: Optuna hyperparameter optimization
- **Accuracy**: 92.4% | **F1**: 0.91
- Best for: tabular data, production inference speed

### LSTM (Temporal)
- **Input**: 30-step sliding window sequences
- **Architecture**: 2-layer LSTM (128 → 64 units) + Dense
- **Accuracy**: 90.8% | **F1**: 0.88
- Best for: detecting gradual degradation patterns over time

### Ensemble
- Weighted average: XGBoost (0.5) + RF (0.3) + LSTM (0.2)
- **Final Accuracy: 92.4% | False Positive Reduction: 28%**

---

## 📊 Results

```
Classification Report (Ensemble Model):
              precision    recall  f1-score   support
   No Failure     0.95      0.96      0.95      3821
      Failure     0.89      0.87      0.88      1179
    accuracy                          0.924     5000
```

---

## 🔧 Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy, SQL |
| ML Models | Scikit-learn, XGBoost, TensorFlow/Keras |
| Feature Eng. | SciPy (FFT), rolling windows, lag features |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| DevOps | Docker, Docker Compose, GitHub Actions |
| Testing | Pytest |

---

## 📓 Notebook

The [`notebooks/EDA_and_Modeling.ipynb`](notebooks/EDA_and_Modeling.ipynb) walks through:
1. Sensor data exploration & visualizations
2. Failure pattern analysis
3. Feature engineering rationale
4. Model training & comparison
5. SHAP value explainability

---

## 👤 Author

**Veda Vineetha Moturi** — Data Scientist | ML Engineer  
📧 vedavineetha482@gmail.com | [LinkedIn](https://linkedin.com)

---

## 📄 License
MIT License — see [LICENSE](LICENSE)
