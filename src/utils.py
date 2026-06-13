import numpy as np
import pandas as pd


def gini_coefficient(actual, predicted):
    """
    Computes the Gini coefficient — standard actuarial discrimination metric.
    Gini = 2 * AUC - 1, measures how well the model separates high vs low risk.
    Range: 0 (random) to 1 (perfect).
    """
    df = pd.DataFrame({"actual": np.array(actual), "predicted": np.array(predicted)})
    df = df.sort_values("predicted", ascending=False).reset_index(drop=True)

    cum_actual = df["actual"].cumsum() / df["actual"].sum()
    cum_population = (df.index + 1) / len(df)

    # Manual trapezoidal integration — no numpy version dependency
    auc = float(np.sum(
        (cum_population.values[1:] - cum_population.values[:-1]) *
        (cum_actual.values[1:] + cum_actual.values[:-1]) / 2
    ))
    return round(abs(2 * auc - 1), 4)