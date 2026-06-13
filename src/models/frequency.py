import numpy as np
import pandas as pd
import statsmodels.api as sm


def train_frequency_model(X_train, y_freq_train, exp_train):
    """
    Fits a Poisson GLM for claim frequency.
    Exposure is included as an offset (log scale).
    """
    print("Training Poisson GLM...")

    X_train_const = sm.add_constant(X_train)
    offset = np.log(exp_train.clip(lower=1e-6))

    model = sm.GLM(
        y_freq_train,
        X_train_const,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=offset
    )

    result = model.fit(maxiter=100, tol=1e-8)
    print(result.summary())
    return result


def predict_frequency(result, X, exposure):
    """
    Returns predicted claim frequency for a feature matrix X.
    """
    X_const = sm.add_constant(X, has_constant="add")
    offset = np.log(exposure.clip(lower=1e-6))
    return result.predict(X_const, offset=offset)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.data_loader import load_mtpl_data
    from src.features import engineer_features

    df = load_mtpl_data()
    (X_train, X_test,
     y_freq_train, y_freq_test,
     y_sev_train, y_sev_test,
     exp_train, exp_test,
     feature_cols) = engineer_features(df)

    result = train_frequency_model(X_train, y_freq_train, exp_train)

    preds = predict_frequency(result, X_test, exp_test)
    print(f"\nSample frequency predictions: {preds[:5].values}")