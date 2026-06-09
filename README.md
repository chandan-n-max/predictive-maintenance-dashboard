# 🔧 Predictive Maintenance Dashboard
### NASA CMAPSS FD001 — Turbofan Engine Fleet Monitoring

**Live Demo:** https://predictive-maintenance-dashboard-t36obbuhq8inatabgrgmnk.streamlit.app

---

## What This Project Does

Most predictive maintenance projects stop at a number — "Engine 3 will fail in 48 cycles." This project goes further. It tells you **what to do about it**.

Built on NASA's CMAPSS FD001 turbofan engine dataset, this dashboard predicts Remaining Useful Life (RUL) for 100 engines and converts that prediction into an actionable maintenance recommendation — including which specific sensor is behaving abnormally and why it matters.

---

## The Unique Layer

**Sensor Anomaly Flagging** — For each engine, the dashboard identifies the most anomalous sensor by calculating how far it has drifted from that engine's own historical baseline (z-score). This is not a global average comparison — it uses each engine's individual degradation profile.

Output example:
> *Engine 34 — CRITICAL. Predicted RUL: 6.6 cycles. Primary anomaly: sensor12 (Fuel flow ratio) — 3.584 standard deviations above baseline. Recommended action: Schedule urgent inspection immediately.*

---

## Tech Stack

- **Data:** NASA CMAPSS FD001 (100 training engines, 20,631 rows, 21 sensors)
- **Feature Engineering:** Rolling mean and std over 5 and 10 cycle windows (44 features)
- **Models:** Random Forest (RMSE: 19.97) and XGBoost (RMSE: 19.60, R²: 0.777)
- **Dashboard:** Streamlit + Plotly
- **Deployment:** Streamlit Community Cloud

---

## Dashboard Features

- Engine selector — view any of the 100 test engines
- Predicted RUL with actual RUL comparison
- Risk badge — Critical / Warning / Stable
- Sensor trend chart — last 30 cycles with rolling mean
- Maintenance recommendation card — flagged sensor, z-score, action, downtime risk
- Fleet overview table — all 100 engines ranked by risk
- Fleet risk distribution charts

---

## Model Performance

| Model | RMSE | R² |
|---|---|---|
| Random Forest | 19.97 cycles | 0.769 |
| XGBoost | 19.60 cycles | 0.777 |

The model is strongest in the critical RUL zone (0–30 cycles) — exactly where accuracy matters most for maintenance decisions.

---

## Fleet Insights

- 20 engines in Critical zone (RUL < 30)
- 20 engines in Warning zone (RUL 30–80)
- 60 engines Stable (RUL > 80)
- sensor14 (Corrected fan speed) flagged as anomalous in 32/100 engines — suggesting a systemic fleet-wide degradation pattern

---

## About

Built by **Chandan N** — PLM Technical Consultant (Tata Technologies, Dassault Systèmes) and MS Computer Science (AI/ML) candidate. This project bridges manufacturing domain expertise with data science — the sensor flagging logic is only possible with knowledge of what these sensors physically measure in a turbofan engine.

**LinkedIn:** https://www.linkedin.com/in/chandan-n-1730b81a0/