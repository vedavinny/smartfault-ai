"""
SmartFault AI — FastAPI REST Microservice
Real-time predictive maintenance inference endpoint.

Run: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.schemas import (
    SensorReading, BatchSensorReading,
    PredictionResponse, BatchPredictionResponse, HealthResponse
)

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SmartFault AI — Predictive Maintenance API",
    description="""
## ⚙️ SmartFault AI

Real-time machinery failure prediction using multi-sensor IoT data.

### Features
- **Single prediction**: POST `/predict` with live sensor readings
- **Batch prediction**: POST `/predict/batch` for multiple machines
- **Machine history**: GET `/history/{machine_id}`
- **Model metrics**: GET `/metrics`

### Model
Weighted ensemble of Random Forest, XGBoost, and LSTM with **92.4% accuracy**.
    """,
    version="1.0.0",
    contact={"name": "Veda Vineetha Moturi", "email": "vedavineetha482@gmail.com"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory history store (replace with DB in production)
prediction_history: dict = {}

# Lazy-load the predictor
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            from models.ensemble import EnsemblePredictor
            _predictor = EnsemblePredictor()
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}. Run training scripts first.")
    return _predictor


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    return {
        "service": "SmartFault AI Predictive Maintenance",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    try:
        predictor = get_predictor()
        models_loaded = list(predictor.models.keys())
        status = "healthy"
    except Exception as e:
        models_loaded = []
        status = f"degraded: {str(e)}"

    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/metrics", tags=["Model"])
def get_metrics():
    """Returns training metrics for all loaded models."""
    import json, glob
    metrics = {}
    for path in glob.glob("models/*_metrics.json"):
        name = os.path.basename(path).replace("_metrics.json", "")
        with open(path) as f:
            metrics[name] = json.load(f)
    if not metrics:
        return {"message": "No metrics found. Train models first."}
    return metrics


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(reading: SensorReading, background_tasks: BackgroundTasks):
    """
    Predict failure probability for a single machine based on current sensor readings.
    
    Risk levels:
    - **LOW** (<30%): Normal operation
    - **MEDIUM** (30-60%): Monitor closely
    - **HIGH** (60-80%): Schedule maintenance soon
    - **CRITICAL** (>80%): Immediate action required
    """
    try:
        predictor = get_predictor()
        result = predictor.predict_single(
            machine_id=reading.machine_id,
            temperature=reading.temperature,
            vibration=reading.vibration,
            pressure=reading.pressure,
            rpm=reading.rpm,
        )
        result["timestamp"] = datetime.utcnow().isoformat()

        # Store in history (background)
        def store_history():
            mid = reading.machine_id
            if mid not in prediction_history:
                prediction_history[mid] = []
            prediction_history[mid].append(result)
            prediction_history[mid] = prediction_history[mid][-100:]  # keep last 100

        background_tasks.add_task(store_history)
        return PredictionResponse(**result)

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(payload: BatchSensorReading):
    """Predict failure probability for multiple machines at once."""
    try:
        predictor = get_predictor()
        results = []
        for reading in payload.readings:
            result = predictor.predict_single(
                machine_id=reading.machine_id,
                temperature=reading.temperature,
                vibration=reading.vibration,
                pressure=reading.pressure,
                rpm=reading.rpm,
            )
            result["timestamp"] = datetime.utcnow().isoformat()
            results.append(PredictionResponse(**result))

        high_risk = [r for r in results if r.risk_level in ("HIGH", "CRITICAL")]
        return BatchPredictionResponse(
            total=len(results),
            high_risk_count=len(high_risk),
            predictions=results,
            batch_timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{machine_id}", tags=["History"])
def get_history(machine_id: str, limit: int = 20):
    """Return prediction history for a specific machine (last N predictions)."""
    history = prediction_history.get(machine_id, [])
    if not history:
        raise HTTPException(status_code=404, detail=f"No history for machine '{machine_id}'")
    return {"machine_id": machine_id, "count": len(history), "history": history[-limit:]}


@app.get("/machines", tags=["History"])
def list_machines():
    """List all machines with recorded predictions."""
    return {
        "machines": list(prediction_history.keys()),
        "total": len(prediction_history),
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
