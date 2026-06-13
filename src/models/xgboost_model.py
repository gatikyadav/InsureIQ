import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def train_xgboost_model(X_train, y_freq_train, y_sev_train, exp_train):
    """
    Trains an XGBoost model with Tweedie loss to predict pure premium directly.
    Tweedie is appropriate for compound Poisson-Gamma distributions (insurance losses).
    """
    print("Training XGBoost (Tweedie loss)...")

    # Pure premium = total loss per unit exposure
    y_pure_premium = y_sev_train / exp_train.clip(lower=1e-6)

    # Sample weights = exposure (standard actuarial weighting)
    weights = exp_train.clip(lower=1e-6)

    model = xgb.XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=1.5,   # between Poisson (1) and Gamma (2)
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_pure_premium,
        sample_weight=weights,
        eval_set=[(X_train, y_pure_premium)],
        verbose=100
    )

    print("XGBoost training complete.")
    return model


def predict_xgboost(model, X):
    """
    Returns predicted pure premium for a feature matrix X.
    """
    return model.predict(X)


def compare_models(glm_preds, xgb_preds, y_freq_test, y_sev_test, exp_test):
    """
    Compares GLM and XGBoost predictions on the test set.
    Returns a summary dict.
    """
    # Actual pure premium
    actual = y_sev_test / exp_test.clip(lower=1e-6)

    glm_mse = mean_squared_error(actual, glm_preds)
    xgb_mse = mean_squared_error(actual, xgb_preds)

    print(f"\n--- Model Comparison ---")
    print(f"GLM  MSE: {glm_mse:,.2f}")
    print(f"XGB  MSE: {xgb_mse:,.2f}")
    print(f"XGBoost improvement: {(glm_mse - xgb_mse) / glm_mse * 100:.1f}%")

    return {
        "glm_mse": glm_mse,
        "xgb_mse": xgb_mse,
        "actual": actual,
        "glm_preds": glm_preds,
        "xgb_preds": xgb_preds
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.data_loader import load_mtpl_data
    from src.features import engineer_features
    from src.models.frequency import train_frequency_model, predict_frequency
    from src.models.severity import train_severity_model, predict_severity

    df = load_mtpl_data()
    (X_train, X_test,
     y_freq_train, y_freq_test,
     y_sev_train, y_sev_test,
     exp_train, exp_test,
     feature_cols) = engineer_features(df)

    freq_result = train_frequency_model(X_train, y_freq_train, exp_train)
    sev_result = train_severity_model(X_train, y_sev_train, y_freq_train)

    xgb_model = train_xgboost_model(X_train, y_freq_train, y_sev_train, exp_train)

    freq_preds = predict_frequency(freq_result, X_test, exp_test)
    sev_preds = predict_severity(sev_result, X_test)
    glm_pure_premium = freq_preds * sev_preds

    xgb_pure_premium = predict_xgboost(xgb_model, X_test)

    results = compare_models(
        glm_pure_premium, xgb_pure_premium,
        y_freq_test, y_sev_test, exp_test
    )