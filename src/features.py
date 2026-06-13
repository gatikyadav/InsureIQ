import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def engineer_features(df: pd.DataFrame):
    """
    Cleans and encodes the raw MTPL DataFrame for modeling.
    Returns X_train, X_test, y_freq_train, y_freq_test,
                            y_sev_train, y_sev_test,
                            exposure_train, exposure_test
    """
    data = df.copy()

    # --- Clean VehGas (has stray quotes) ---
    data["VehGas"] = data["VehGas"].str.replace("'", "").str.strip()

    # --- Encode categoricals as dummies ---
    cat_cols = ["Area", "VehBrand", "VehGas", "Region"]
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    # --- Cap BonusMalus outliers ---
    data["BonusMalus"] = data["BonusMalus"].clip(upper=150)

    # --- Log-transform Density (highly skewed) ---
    data["LogDensity"] = np.log1p(data["Density"])
    data = data.drop(columns=["Density"])

    # --- Feature columns ---
    drop_cols = ["IDpol", "ClaimNb", "ClaimAmount", "Exposure"]
    feature_cols = [c for c in data.columns if c not in drop_cols]

    X = data[feature_cols].astype(float)
    y_freq = data["ClaimNb"]
    y_sev = data["ClaimAmount"]
    exposure = data["Exposure"]

    # --- Train/test split (stratify on ClaimNb > 0) ---
    strat = (y_freq > 0).astype(int)
    X_train, X_test, \
    y_freq_train, y_freq_test, \
    y_sev_train, y_sev_test, \
    exp_train, exp_test = train_test_split(
        X, y_freq, y_sev, exposure,
        test_size=0.2,
        random_state=42,
        stratify=strat
    )

    print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    print(f"Features: {X_train.shape[1]}")
    print(f"Claim rate (train): {(y_freq_train > 0).mean():.3%}")

    return (X_train, X_test,
            y_freq_train, y_freq_test,
            y_sev_train, y_sev_test,
            exp_train, exp_test,
            feature_cols)


if __name__ == "__main__":
    from data_loader import load_mtpl_data
    df = load_mtpl_data()
    results = engineer_features(df)