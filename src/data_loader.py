import pandas as pd
from sklearn.datasets import fetch_openml

def load_mtpl_data():
    """
    Loads the French MTPL frequency and severity tables from OpenML,
    merges them, and returns a clean combined DataFrame.
    """
    print("Loading frequency table (678K rows)... this may take ~30s")
    freq = fetch_openml("freMTPL2freq", as_frame=True).frame
    
    print("Loading severity table...")
    sev = fetch_openml("freMTPL2sev", as_frame=True).frame

    # Aggregate severity to one row per policy
    sev_agg = sev.groupby("IDpol")["ClaimAmount"].sum().reset_index()
    sev_agg.columns = ["IDpol", "ClaimAmount"]

    # Merge on policy ID
    freq["IDpol"] = freq["IDpol"].astype(int)
    sev_agg["IDpol"] = sev_agg["IDpol"].astype(int)
    df = freq.merge(sev_agg, on="IDpol", how="left")

    # Policies with no claims get ClaimAmount = 0
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    print(f"Dataset ready: {df.shape[0]:,} policies, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    df = load_mtpl_data()
    print(df.head())
    print(df.dtypes)