# ── SECTION 1: Imports and Page Config ───────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

# Page config — must be the very first Streamlit command
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SECTION 2: Load Data and Models ──────────────────────

@st.cache_data
def load_data():
    fleet_df     = pd.read_csv('fleet_recommendations.csv')
    train_df     = pd.read_csv('train_processed.csv')
    test_df      = pd.read_csv('test_predictions.csv')
    return fleet_df, train_df, test_df

@st.cache_resource
def load_model():
    model        = joblib.load('xgb_model.pkl')
    scaler       = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return model, scaler, feature_cols

fleet_df, train_df, test_df = load_data()
model, scaler, feature_cols = load_model()

# ── SECTION 3: Sidebar ────────────────────────────────────

st.sidebar.title("🔧 Fleet Control Panel")
st.sidebar.markdown("---")

# Engine selector
engine_ids = sorted(fleet_df['unit_id'].tolist())
selected_engine = st.sidebar.selectbox(
    "Select Engine Unit",
    engine_ids,
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Fleet Summary")

# Risk counts
critical_count = len(fleet_df[fleet_df['risk_level'] == 'CRITICAL'])
warning_count  = len(fleet_df[fleet_df['risk_level'] == 'WARNING'])
stable_count   = len(fleet_df[fleet_df['risk_level'] == 'STABLE'])

st.sidebar.error(f"🔴 Critical : {critical_count} engines")
st.sidebar.warning(f"🟡 Warning  : {warning_count} engines")
st.sidebar.success(f"🟢 Stable   : {stable_count} engines")

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: NASA CMAPSS FD001")
st.sidebar.caption("Model: XGBoost | RMSE: 19.60 cycles")
st.sidebar.caption("Built by Chandan N")

# ── SECTION 4: Main Header ────────────────────────────────

st.title("🔧 Predictive Maintenance Dashboard")
st.markdown("**NASA CMAPSS FD001 — Turbofan Engine Fleet Monitoring**")
st.markdown("---")

# ── SECTION 5: Engine Detail Cards ───────────────────────

# Get selected engine data
engine_data = fleet_df[fleet_df['unit_id'] == selected_engine].iloc[0]

predicted_rul  = engine_data['predicted_rul']
actual_rul     = engine_data['actual_rul']
risk_level     = engine_data['risk_level']
action         = engine_data['action']
flagged_sensor = engine_data['flagged_sensor']
sensor_meaning = engine_data['sensor_meaning']
z_score        = engine_data['z_score']
downtime_risk  = engine_data['downtime_risk']

st.subheader(f"Engine Unit {selected_engine} — Detail View")

# Three metric cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Predicted RUL",
        value=f"{predicted_rul} cycles",
        delta=f"Actual: {actual_rul} cycles"
    )

with col2:
    if risk_level == 'CRITICAL':
        st.error(f"🔴 {risk_level}")
    elif risk_level == 'WARNING':
        st.warning(f"🟡 {risk_level}")
    else:
        st.success(f"🟢 {risk_level}")

with col3:
    st.info(f"📋 {action}")

# ── SECTION 6: Sensor Trend Chart ────────────────────────

st.markdown("---")
st.subheader(f"Sensor Trends — Last 30 Cycles")

# Get training data for a similar engine as proxy
# Use engine with same unit_id if exists, else engine 1
engine_history = train_df[train_df['unit'] == selected_engine]
if len(engine_history) == 0:
    engine_history = train_df[train_df['unit'] == 1]

# Take last 30 cycles
last_30 = engine_history.tail(30).copy()

# Sensors to show — top ones from feature importance
display_sensors = ['sensor4', 'sensor11', 'sensor9',
                   'sensor12', 'sensor14', 'sensor2']

# Let user pick which sensor to focus on
selected_sensor = st.selectbox(
    "Select sensor to view trend",
    display_sensors,
    index=0
)

# Plot selected sensor trend
fig_trend = go.Figure()

fig_trend.add_trace(go.Scatter(
    x=last_30['cycle'],
    y=last_30[selected_sensor],
    mode='lines+markers',
    name=selected_sensor,
    line=dict(color='#2196F3', width=2),
    marker=dict(size=5)
))

# Add rolling mean
rolling_mean = last_30[selected_sensor].rolling(5).mean()
fig_trend.add_trace(go.Scatter(
    x=last_30['cycle'],
    y=rolling_mean,
    mode='lines',
    name='Rolling Mean (5)',
    line=dict(color='#FF9800', width=2, dash='dash')
))

fig_trend.update_layout(
    title=f'{selected_sensor} — Last 30 Cycles (Engine {selected_engine})',
    xaxis_title='Cycle',
    yaxis_title='Sensor Value',
    height=350,
    legend=dict(orientation='h', y=1.1),
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig_trend, use_container_width=True)

# ── SECTION 7: Maintenance Recommendation Card ───────────

st.markdown("---")
st.subheader("🔍 Maintenance Recommendation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Primary Anomalous Sensor**")
    st.markdown(f"🔧 `{flagged_sensor}` — {sensor_meaning}")
    st.markdown(f"**Deviation from baseline:** {z_score} standard deviations")

    if z_score > 3:
        st.error("⚠️ Highly anomalous — immediate inspection of this component recommended")
    elif z_score > 2:
        st.warning("⚠️ Moderately anomalous — include in next inspection")
    else:
        st.info("ℹ️ Mildly anomalous — monitor trend")

with col2:
    st.markdown("**Downtime Risk Assessment**")
    st.markdown(f"📊 {downtime_risk}")
    st.markdown("**Recommended Action**")
    st.markdown(f"✅ {action}")

# ── SECTION 8: Fleet Overview Table ──────────────────────

st.markdown("---")
st.subheader("📋 Fleet Overview — All 100 Engines Ranked by Risk")

# Colour code risk level
def highlight_risk(val):
    if val == 'CRITICAL':
        return 'background-color: #ffcccc'
    elif val == 'WARNING':
        return 'background-color: #fff3cc'
    else:
        return 'background-color: #ccffcc'

# Display columns
display_cols = [
    'unit_id', 'predicted_rul', 'actual_rul',
    'risk_level', 'action', 'flagged_sensor',
    'sensor_meaning', 'z_score'
]

styled_fleet = fleet_df[display_cols].style.map(
    highlight_risk, subset=['risk_level']
)

st.dataframe(styled_fleet, use_container_width=True, height=400)

# ── SECTION 9: Fleet Risk Distribution Chart ─────────────

st.markdown("---")
st.subheader("📊 Fleet Risk Distribution")

col1, col2 = st.columns(2)

with col1:
    # Risk distribution bar chart
    risk_counts = fleet_df['risk_level'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']

    color_map = {
        'CRITICAL': '#e74c3c',
        'WARNING' : '#f39c12',
        'STABLE'  : '#27ae60'
    }

    fig_risk = px.bar(
        risk_counts,
        x='Risk Level',
        y='Count',
        color='Risk Level',
        color_discrete_map=color_map,
        title='Engines by Risk Level',
        text='Count'
    )
    fig_risk.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_risk, use_container_width=True, key="fig_risk")

with col2:
    # RUL scatter all engines
    fig_scatter = px.scatter(
        fleet_df,
        x='unit_id',
        y='predicted_rul',
        color='risk_level',
        color_discrete_map=color_map,
        title='Predicted RUL — All Engines',
        labels={'unit_id': 'Engine ID', 'predicted_rul': 'Predicted RUL (cycles)'}
    )
    fig_scatter.add_hline(y=30, line_dash='dash',
                          line_color='red', annotation_text='Critical (30)')
    fig_scatter.add_hline(y=80, line_dash='dash',
                          line_color='orange', annotation_text='Warning (80)')
    fig_scatter.update_layout(height=350)
    st.plotly_chart(fig_scatter, use_container_width=True, key="fig_scatter")


# ── SECTION 10: Footer ────────────────────────────────────

st.markdown("---")
st.caption("Built by Chandan N | PLM Technical Consultant & ML Engineer")
st.caption("Dataset: NASA CMAPSS FD001 | Model: XGBoost | RMSE: 19.60 cycles | R²: 0.7775")

