"""
SmartFault AI — Real-Time Monitoring Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
import random

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartFault AI — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: #1a1f2e; border-radius: 10px; padding: 10px; border-left: 4px solid #00b4d8; }
    .risk-critical { color: #ff4b4b; font-weight: bold; font-size: 1.4rem; }
    .risk-high { color: #ff8c00; font-weight: bold; }
    .risk-medium { color: #ffd60a; font-weight: bold; }
    .risk-low { color: #06d6a0; font-weight: bold; }
    h1 { color: #00b4d8; }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/gear.png", width=60)
    st.title("⚙️ SmartFault AI")
    st.caption("Predictive Maintenance Dashboard")
    st.divider()

    machine_ids = [f"MACHINE_{i:03d}" for i in range(1, 11)]
    selected_machine = st.selectbox("Select Machine", machine_ids)
    refresh_rate = st.slider("Refresh Rate (seconds)", 2, 30, 5)
    auto_refresh = st.toggle("Auto Refresh", value=True)
    demo_mode = st.toggle("Demo Mode (no API needed)", value=True)

    st.divider()
    st.markdown("**Model Performance**")
    st.markdown("- XGBoost: 92.4% acc")
    st.markdown("- Random Forest: 91.2% acc")
    st.markdown("- LSTM: 90.8% acc")
    st.markdown("- **Ensemble: 92.4% acc**")


# ─────────────────────────────────────────────
# Data / API helpers
# ─────────────────────────────────────────────
def get_risk_color(risk: str) -> str:
    return {"LOW": "#06d6a0", "MEDIUM": "#ffd60a", "HIGH": "#ff8c00", "CRITICAL": "#ff4b4b"}.get(risk, "#888")

def simulate_reading(machine_id: str, step: int, inject_failure: bool = False):
    """Generate realistic synthetic sensor data for demo mode."""
    base_temp = 68 + hash(machine_id) % 8
    ramp = min(1.0, step / 50) if inject_failure else 0
    noise = lambda s: np.random.normal(0, s)
    return {
        "machine_id": machine_id,
        "temperature": round(base_temp + noise(1.5) + ramp * random.uniform(15, 25), 2),
        "vibration": round(0.25 + noise(0.03) + ramp * random.uniform(0.4, 0.8), 3),
        "pressure": round(107 + noise(2) - ramp * random.uniform(8, 20), 2),
        "rpm": round(1440 + noise(20) + ramp * random.uniform(-200, 30), 1),
    }

def call_predict(reading: dict) -> dict:
    if demo_mode:
        # Simulate API response
        temp_dev = (reading["temperature"] - 68) / 30
        vib_dev = (reading["vibration"] - 0.25) / 0.8
        pres_dev = (107 - reading["pressure"]) / 20
        prob = float(np.clip(0.1 * temp_dev + 0.5 * vib_dev + 0.4 * pres_dev + random.uniform(-0.05, 0.05), 0, 1))
        risk = "CRITICAL" if prob > 0.9 else "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
        hours = round(max(1, (1 - prob) * 24), 1) if prob > 0.3 else None
        actions = {
            "LOW": "Monitor normally.",
            "MEDIUM": "Flag for inspection within 48 hours.",
            "HIGH": "Schedule maintenance within 6-12 hours.",
            "CRITICAL": "Immediate shutdown recommended!"
        }
        return {
            "failure_probability": round(prob, 4), "risk_level": risk,
            "predicted_failure_in_hours": hours, "recommended_action": actions[risk],
            "models_used": ["xgboost", "rf", "lstm"], "confidence": round(min(1, prob + 0.05), 3),
        }
    try:
        resp = requests.post(f"{API_URL}/predict", json=reading, timeout=5)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "step" not in st.session_state:
    st.session_state.step = 0
if "inject_failure" not in st.session_state:
    st.session_state.inject_failure = False


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("⚙️ SmartFault AI — Predictive Maintenance Dashboard")
st.caption(f"Machine: **{selected_machine}** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Failure injection button
col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
with col_h2:
    if st.button("🔴 Inject Failure Pattern", use_container_width=True):
        st.session_state.inject_failure = True
        st.session_state.step = 0
with col_h3:
    if st.button("✅ Reset to Normal", use_container_width=True):
        st.session_state.inject_failure = False
        st.session_state.step = 0


# ─────────────────────────────────────────────
# Live Reading & Prediction
# ─────────────────────────────────────────────
st.session_state.step += 1
reading = simulate_reading(selected_machine, st.session_state.step, st.session_state.inject_failure)
pred = call_predict(reading)

if "error" not in pred:
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append({**reading, **pred, "ts": ts})
    st.session_state.history = st.session_state.history[-60:]

hist_df = pd.DataFrame(st.session_state.history)

# ─────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────
risk = pred.get("risk_level", "N/A")
prob = pred.get("failure_probability", 0)
hours = pred.get("predicted_failure_in_hours")
color = get_risk_color(risk)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("🌡️ Temperature", f"{reading['temperature']} °C", delta=f"{reading['temperature'] - 68:.1f}")
with k2:
    st.metric("📳 Vibration", f"{reading['vibration']} g", delta=f"{reading['vibration'] - 0.25:.3f}")
with k3:
    st.metric("🔵 Pressure", f"{reading['pressure']} bar", delta=f"{reading['pressure'] - 107:.1f}")
with k4:
    st.metric("⚡ RPM", f"{reading['rpm']:.0f}", delta=f"{reading['rpm'] - 1440:.0f}")
with k5:
    st.metric("🎯 Failure Risk", f"{prob*100:.1f}%", delta=risk)

st.divider()

# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────
chart_col, gauge_col = st.columns([3, 1])

with chart_col:
    if len(hist_df) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df["ts"], y=hist_df["failure_probability"] * 100,
            name="Failure Risk %", line=dict(color="#ff6b6b", width=2.5), fill="tozeroy",
            fillcolor="rgba(255,107,107,0.15)"
        ))
        fig.add_trace(go.Scatter(
            x=hist_df["ts"], y=hist_df["temperature"],
            name="Temperature (°C)", line=dict(color="#00b4d8", width=1.5),
            yaxis="y2", opacity=0.7
        ))
        fig.update_layout(
            title="Failure Risk & Temperature Over Time",
            paper_bgcolor="#0e1117", plot_bgcolor="#131720",
            font=dict(color="#ccc"), height=280,
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Risk %", range=[0, 100], gridcolor="#222"),
            yaxis2=dict(title="°C", overlaying="y", side="right", gridcolor="#222"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

with gauge_col:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": f"Risk<br><span style='font-size:1rem;color:{color}'>{risk}</span>", "font": {"color": "#ccc"}},
        number={"suffix": "%", "font": {"color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555"},
            "bar": {"color": color},
            "bgcolor": "#1a1f2e",
            "steps": [
                {"range": [0, 30], "color": "#0d1b2a"},
                {"range": [30, 60], "color": "#1a2035"},
                {"range": [60, 80], "color": "#1f2040"},
                {"range": [80, 100], "color": "#2a1020"},
            ],
            "threshold": {"line": {"color": "#ff4b4b", "width": 3}, "value": 80},
        }
    ))
    gauge.update_layout(
        paper_bgcolor="#0e1117", font={"color": "#ccc"},
        height=250, margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(gauge, use_container_width=True)

# ─────────────────────────────────────────────
# Multi-sensor Chart
# ─────────────────────────────────────────────
if len(hist_df) > 2:
    tab1, tab2 = st.tabs(["📈 Sensor Trends", "📊 All Machines"])
    with tab1:
        fig2 = go.Figure()
        for sensor, color_s in [("vibration", "#f9c74f"), ("pressure", "#90be6d")]:
            fig2.add_trace(go.Scatter(
                x=hist_df["ts"], y=hist_df[sensor],
                name=sensor.title(), line=dict(color=color_s, width=1.5)
            ))
        fig2.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#131720", font=dict(color="#ccc"),
            height=200, margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#222"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        # Simulate fleet overview
        fleet_data = []
        for mid in machine_ids:
            r = simulate_reading(mid, random.randint(1, 30))
            p = call_predict(r)
            fleet_data.append({"Machine": mid, "Risk": p.get("failure_probability", 0) * 100, "Level": p["risk_level"]})
        fleet_df = pd.DataFrame(fleet_data)
        fig3 = px.bar(
            fleet_df, x="Machine", y="Risk", color="Level",
            color_discrete_map={"LOW": "#06d6a0", "MEDIUM": "#ffd60a", "HIGH": "#ff8c00", "CRITICAL": "#ff4b4b"},
            title="Fleet Risk Overview"
        )
        fig3.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#131720", font=dict(color="#ccc"), height=300)
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# Action Panel
# ─────────────────────────────────────────────
st.divider()
action_col, detail_col = st.columns([1, 1])

with action_col:
    st.markdown(f"### 🔔 Recommended Action")
    action_text = pred.get("recommended_action", "")
    border_color = get_risk_color(risk)
    st.markdown(f"""
    <div style='border-left: 5px solid {border_color}; padding: 12px 20px; 
                background: #1a1f2e; border-radius: 6px; margin-top: 8px;'>
        <span style='font-size: 1.05rem;'>{action_text}</span>
        {f"<br><br><b>⏱️ Estimated time to failure: {hours} hours</b>" if hours else ""}
    </div>
    """, unsafe_allow_html=True)

with detail_col:
    st.markdown("### 📋 Prediction Details")
    st.json({
        "failure_probability": f"{prob*100:.2f}%",
        "risk_level": risk,
        "models_used": pred.get("models_used", []),
        "confidence": f"{pred.get('confidence', 0)*100:.1f}%",
        "last_updated": datetime.now().strftime("%H:%M:%S"),
    })

# ─────────────────────────────────────────────
# Auto Refresh
# ─────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
