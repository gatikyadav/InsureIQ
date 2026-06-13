import numpy as np
import pandas as pd
import statsmodels.api as sm


def train_severity_model(X_train, y_sev_train, y_freq_train):
    """
    Fits a Gamma GLM for claim severity.
    Only trained on policies with at least one claim (ClaimNb > 0).
    """
    print("Training Gamma GLM...")

    # Filter to claims-only rows
    mask = y_freq_train > 0
    X_claims = X_train[mask]
    y_claims = y_sev_train[mask]

    # Drop zero or negative claim amounts (data quality)
    valid = y_claims > 0
    X_claims = X_claims[valid]
    y_claims = y_claims[valid]

    print(f"Training on {X_claims.shape[0]:,} claims-only rows")

    X_const = sm.add_constant(X_claims)

    model = sm.GLM(
        y_claims,
        X_const,
        family=sm.families.Gamma(link=sm.families.links.Log())
    )

    result = model.fit(maxiter=100, tol=1e-8)
    print(result.summary())
    return result


def predict_severity(result, X):
    """
    Returns predicted average claim severity for a feature matrix X.
    """
    X_const = sm.add_constant(X, has_constant="add")
    return result.predict(X_const)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.data_loader import load_mtpl_data
    from src.features import engineer_features
    from src.models.frequency import train_frequency_model, predict_frequency

    df = load_mtpl_data()
    (X_train, X_test,
     y_freq_train, y_freq_test,
     y_sev_train, y_sev_test,
     exp_train, exp_test,
     feature_cols) = engineer_features(df)

    freq_result = train_frequency_model(X_train, y_freq_train, exp_train)
    sev_result = train_severity_model(X_train, y_sev_train, y_freq_train)

    freq_preds = predict_frequency(freq_result, X_test, exp_test)
    sev_preds = predict_severity(sev_result, X_test)

    pure_premium = freq_preds * sev_preds
    print(f"\nSample pure premiums: {pure_premium[:5].values}")