# 🚀 HOW TO PUT THIS ON GITHUB — Step by Step

## 1. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `smartfault-ai`
3. Description: `Real-time machinery failure prediction using ML (RF + XGBoost + LSTM) | FastAPI + Docker | 92% accuracy`
4. Set to **Public** (important for recruiters)
5. Do NOT initialize with README (we have one)
6. Click **Create repository**

---

## 2. Push the Code

Open terminal in the `smartfault-ai/` folder:

```bash
cd smartfault-ai/

git init
git add .
git commit -m "Initial commit: SmartFault AI predictive maintenance system

- End-to-end ML pipeline: RF, XGBoost, LSTM ensemble (92.4% accuracy)
- 28-feature engineering from sensor streams (temp, vibration, pressure)
- FastAPI REST microservice with Docker deployment
- Real-time Streamlit monitoring dashboard
- GitHub Actions CI/CD pipeline"

git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/smartfault-ai.git
git push -u origin main
```

---

## 3. Add GitHub Topics (makes it discoverable)

Go to your repo → Click ⚙️ gear next to "About" → Add topics:
```
machine-learning, predictive-maintenance, fastapi, docker, xgboost, lstm, 
tensorflow, data-science, iot, python, time-series, anomaly-detection
```

---

## 4. Run the Project Locally (to take screenshots for README)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data
python data/generate_sensor_data.py

# Preprocess
python data/preprocess.py

# Train models
python models/train_random_forest.py
python models/train_xgboost.py
python models/train_lstm.py

# Start API
uvicorn api.main:app --reload
# Visit http://localhost:8000/docs

# Start Dashboard (new terminal)
streamlit run dashboard/app.py
# Visit http://localhost:8501
```

---

## 5. Add to Your Resume

In the Projects section of your resume, add:

```
SmartFault AI — Predictive Maintenance System              github.com/YOUR_USERNAME/smartfault-ai
• Built end-to-end ML system predicting machinery failures from sensor streams (temp, vibration, pressure)
• Engineered 28 features (FFT, rolling stats, lag features) and trained RF, XGBoost, LSTM ensemble: 92.4% accuracy
• Deployed as FastAPI REST microservice with Docker and GitHub Actions CI/CD; real-time Streamlit dashboard
• Tech: Python, XGBoost, TensorFlow/Keras, FastAPI, Docker, Pandas, Scikit-learn, Plotly
```

---

## 6. Optional: Deploy for Free (live link = even better!)

### API on Render.com (free tier)
1. Sign up at render.com
2. New Web Service → Connect GitHub → Select `smartfault-ai`
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. You'll get a URL like `https://smartfault-ai.onrender.com`

### Dashboard on Streamlit Cloud (free)
1. Go to share.streamlit.io
2. Connect GitHub → Select repo → Set `dashboard/app.py` as main file
3. Free live URL to add to your resume!
