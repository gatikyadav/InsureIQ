import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_openml
import statsmodels.api as sm
import xgboost as xgb
import shap
import sys, os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import load_mtpl_data
from src.features import engineer_features
from src.models.frequency import train_frequency_model, predict_frequency
from src.models.severity import train_severity_model, predict_severity
from src.models.xgboost_model import train_xgboost_model, predict_xgboost, compare_models

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsureIQ",
    page_icon="🛡️",
    layout="wide"
)

# ── Load & cache everything ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MTPL data and training models... (~2 min)")
def load_all():
    df = load_mtpl_data()
    (X_train, X_test,
     y_freq_train, y_freq_test,
     y_sev_train, y_sev_test,
     exp_train, exp_test,
     feature_cols) = engineer_features(df)

    freq_result = train_frequency_model(X_train, y_freq_train, exp_train)
    sev_result  = train_severity_model(X_train, y_sev_train, y_freq_train)
    xgb_model   = train_xgboost_model(X_train, y_freq_train, y_sev_train, exp_train)

    results = compare_models(
        predict_frequency(freq_result, X_test, exp_test) *
        predict_severity(sev_result, X_test),
        predict_xgboost(xgb_model, X_test),
        y_freq_test, y_sev_test, exp_test
    )

    return (df, X_train, X_test,
            y_freq_train, y_freq_test,
            y_sev_train, y_sev_test,
            exp_train, exp_test,
            feature_cols, freq_result,
            sev_result, xgb_model, results)


(df, X_train, X_test,
 y_freq_train, y_freq_test,
 y_sev_train, y_sev_test,
 exp_train, exp_test,
 feature_cols, freq_result,
 sev_result, xgb_model, results) = load_all()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ InsureIQ")
st.markdown("*Actuarial insurance risk modeling — Poisson/Gamma GLM frequency-severity framework*")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Risk Scorer",
    "📊 Model Comparison",
    "🔍 Rating Factor Explorer",
    "📁 Portfolio Simulator"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Risk Scorer
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Risk Scorer")
    st.markdown("Input policyholder features to get a pure premium estimate.")

    col1, col2 = st.columns(2)

    with col1:
        area        = st.selectbox("Area Code", ["A","B","C","D","E","F"])
        veh_power   = st.slider("Vehicle Power", 4, 15, 7)
        veh_age     = st.slider("Vehicle Age (years)", 0, 20, 5)
        drv_age     = st.slider("Driver Age", 18, 100, 35)

    with col2:
        bonus_malus = st.slider("Bonus Malus", 50, 150, 50)
        density     = st.number_input("Population Density", 1, 30000, 1000)
        veh_gas     = st.selectbox("Fuel Type", ["Diesel", "Regular"])
        veh_brand   = st.selectbox("Vehicle Brand", [
            "B1","B2","B3","B4","B5","B6","B10","B11","B12","B13","B14"
        ])

    if st.button("Calculate Risk", type="primary"):
        # Build input row matching training feature columns
        input_dict = {c: 0.0 for c in feature_cols}
        input_dict["VehPower"]   = veh_power
        input_dict["VehAge"]     = veh_age
        input_dict["DrivAge"]    = drv_age
        input_dict["BonusMalus"] = min(bonus_malus, 150)
        input_dict["LogDensity"] = np.log1p(density)

        # One-hot flags
        for a in ["B","C","D","E","F"]:
            key = f"Area_{a}"
            if key in input_dict:
                input_dict[key] = 1.0 if area == a else 0.0

        for b in ["B2","B3","B4","B5","B6","B10","B11","B12","B13","B14"]:
            key = f"VehBrand_{b}"
            if key in input_dict:
                input_dict[key] = 1.0 if veh_brand == b else 0.0

        if "VehGas_Regular" in input_dict:
            input_dict["VehGas_Regular"] = 1.0 if veh_gas == "Regular" else 0.0

        X_input    = pd.DataFrame([input_dict])
        exp_input  = pd.Series([1.0])  # 1 full policy year

        freq_pred  = predict_frequency(freq_result, X_input, exp_input)[0]
        sev_pred   = predict_severity(sev_result, X_input)[0]
        pure_prem  = freq_pred * sev_pred
        xgb_pred   = predict_xgboost(xgb_model, X_input)[0]

        # Risk tier
        if pure_prem < 50:
            tier, color = "🟢 Low", "green"
        elif pure_prem < 150:
            tier, color = "🟡 Medium", "orange"
        elif pure_prem < 300:
            tier, color = "🔴 High", "red"
        else:
            tier, color = "🚨 Very High", "darkred"

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Claim Frequency", f"{freq_pred:.4f}")
        m2.metric("Avg Severity (€)", f"{sev_pred:,.0f}")
        m3.metric("Pure Premium (€)", f"{pure_prem:,.2f}")
        m4.metric("XGBoost Estimate (€)", f"{xgb_pred:,.2f}")

        st.markdown(f"### Risk Tier: {tier}")

        # Confidence interval (±20% bootstrap approximation)
        lo, hi = pure_prem * 0.80, pure_prem * 1.20
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pure_prem,
            title={"text": "Pure Premium (€)"},
            gauge={
                "axis": {"range": [0, max(500, pure_prem * 1.5)]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0, 50],   "color": "lightgreen"},
                    {"range": [50, 150], "color": "lightyellow"},
                    {"range": [150, 300],"color": "lightsalmon"},
                    {"range": [300, max(500, pure_prem * 1.5)], "color": "lightcoral"},
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"95% CI approximation: €{lo:,.2f} – €{hi:,.2f}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Comparison — GLM vs XGBoost")

    if results is None:
        st.info("Models are still loading...")
        st.stop()

    actual    = np.array(results["actual"])
    glm_preds = np.array(results["glm_preds"])
    xgb_preds = np.array(results["xgb_preds"])

    c1, c2 = st.columns(2)
    c1.metric("GLM MSE",     f"{results['glm_mse']:,.0f}")
    c2.metric("XGBoost MSE", f"{results['xgb_mse']:,.0f}")

    # Lift curves by decile
    comparison_df = pd.DataFrame({
        "actual":   np.array(actual),
        "glm":      np.array(glm_preds),
        "xgb":      np.array(xgb_preds)
    })
    comparison_df["decile"] = pd.qcut(comparison_df["glm"], 10, labels=False)
    lift = comparison_df.groupby("decile")[["actual","glm","xgb"]].mean().reset_index()

    fig2 = px.line(lift, x="decile",
                   y=["actual","glm","xgb"],
                   labels={"value":"Mean Pure Premium (€)", "decile":"Risk Decile"},
                   title="Lift Curve — Actual vs GLM vs XGBoost")
    st.plotly_chart(fig2, use_container_width=True)

    # Largest disagreements
    st.subheader("Top 10 Largest Disagreements")
    comparison_df["disagreement"] = np.abs(comparison_df["glm"] - comparison_df["xgb"])
    top10 = comparison_df.nlargest(10, "disagreement")[["actual","glm","xgb","disagreement"]]
    top10.columns = ["Actual (€)","GLM (€)","XGBoost (€)","|Disagreement| (€)"]
    st.dataframe(top10.style.format("€{:.2f}"), use_container_width=True)

    # Residual plots
    st.subheader("Residuals")
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(glm_preds[:500], actual[:500] - glm_preds[:500], alpha=0.3, s=10)
    axes[0].axhline(0, color="red")
    axes[0].set_title("GLM Residuals")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")

    axes[1].scatter(xgb_preds[:500], actual[:500] - xgb_preds[:500], alpha=0.3, s=10)
    axes[1].axhline(0, color="red")
    axes[1].set_title("XGBoost Residuals")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
    st.pyplot(fig3)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Rating Factor Explorer
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Rating Factor Explorer")
    st.markdown("See how a single feature shifts the GLM pure premium estimate.")

    factor = st.selectbox("Select Feature", ["DrivAge", "VehAge", "BonusMalus", "VehPower"])

    ranges = {
        "DrivAge":    range(18, 90, 2),
        "VehAge":     range(0, 21, 1),
        "BonusMalus": range(50, 151, 5),
        "VehPower":   range(4, 16, 1),
    }

    base = {c: 0.0 for c in feature_cols}
    base["VehGas_Regular"] = 1.0
    base["LogDensity"]     = np.log1p(1000)
    base["BonusMalus"]     = 50
    base["DrivAge"]        = 35
    base["VehAge"]         = 5
    base["VehPower"]       = 7

    records = []
    for val in ranges[factor]:
        row = base.copy()
        row[factor] = val
        X_row = pd.DataFrame([row])
        exp_row = pd.Series([1.0])
        freq = predict_frequency(freq_result, X_row, exp_row)[0]
        sev  = predict_severity(sev_result, X_row)[0]
        records.append({"value": val, "pure_premium": freq * sev})

    factor_df = pd.DataFrame(records)
    fig4 = px.line(factor_df, x="value", y="pure_premium",
                   labels={"value": factor, "pure_premium": "Pure Premium (€)"},
                   title=f"GLM Pure Premium vs {factor}")
    st.plotly_chart(fig4, use_container_width=True)

    # XGBoost partial dependence
    st.subheader("XGBoost Partial Dependence")
    if factor in X_train.columns:
        xgb_records = []
        for val in ranges[factor]:
            row = base.copy()
            row[factor] = val
            X_row = pd.DataFrame([row])
            pred = predict_xgboost(xgb_model, X_row)[0]
            xgb_records.append({"value": val, "pure_premium": pred})
        xgb_df = pd.DataFrame(xgb_records)
        fig5 = px.line(xgb_df, x="value", y="pure_premium",
                       labels={"value": factor, "pure_premium": "Pure Premium (€)"},
                       title=f"XGBoost Partial Dependence — {factor}")
        st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Portfolio Simulator
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Portfolio Simulator")
    st.markdown("Upload a CSV of policyholders to score an entire portfolio.")

    st.markdown("**Required columns:** `Area, VehPower, VehAge, DrivAge, BonusMalus, Density, VehGas, VehBrand`")

    uploaded = st.file_uploader("Upload policyholder CSV", type=["csv"])

    if uploaded:
        portfolio = pd.read_csv(uploaded)
        st.write(f"Loaded {len(portfolio):,} policyholders")

        # Engineer features
        portfolio["VehGas"] = portfolio["VehGas"].str.replace("'","").str.strip()
        cat_cols = ["Area","VehBrand","VehGas","Region"] if "Region" in portfolio.columns else ["Area","VehBrand","VehGas"]
        portfolio_enc = pd.get_dummies(portfolio, columns=cat_cols, drop_first=True)
        portfolio_enc["BonusMalus"] = portfolio_enc["BonusMalus"].clip(upper=150)
        portfolio_enc["LogDensity"] = np.log1p(portfolio_enc["Density"])
        portfolio_enc = portfolio_enc.drop(columns=["Density"], errors="ignore")

        # Align columns
        for c in feature_cols:
            if c not in portfolio_enc.columns:
                portfolio_enc[c] = 0.0
        X_port = portfolio_enc[feature_cols].astype(float)
        exp_port = pd.Series(np.ones(len(X_port)))

        freq_p = predict_frequency(freq_result, X_port, exp_port)
        sev_p  = predict_severity(sev_result, X_port)
        pp     = freq_p * sev_p

        portfolio["PurePremium_GLM"] = pp.values
        portfolio["PurePremium_XGB"] = predict_xgboost(xgb_model, X_port)

        st.subheader("Summary Statistics")
        s1, s2, s3 = st.columns(3)
        s1.metric("Mean Pure Premium (GLM)", f"€{pp.mean():,.2f}")
        s2.metric("Total Expected Loss",     f"€{pp.sum():,.0f}")
        s3.metric("Policies Scored",         f"{len(portfolio):,}")

        fig6 = px.histogram(portfolio, x="PurePremium_GLM", nbins=50,
                            title="Pure Premium Distribution")
        st.plotly_chart(fig6, use_container_width=True)

        csv = portfolio.to_csv(index=False).encode("utf-8")
        st.download_button("Download Scored Portfolio", csv,
                           "scored_portfolio.csv", "text/csv")
    else:
        st.info("No file uploaded yet. Download a sample to get started:")
        sample = pd.DataFrame([{
            "Area": "D", "VehPower": 7, "VehAge": 3,
            "DrivAge": 35, "BonusMalus": 50, "Density": 1000,
            "VehGas": "Diesel", "VehBrand": "B12"
        }] * 5)
        st.download_button("Download Sample CSV",
                           sample.to_csv(index=False).encode("utf-8"),
                           "sample_portfolio.csv", "text/csv")