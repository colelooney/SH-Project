"""
Compute mutual information (MI) between high-level features and the
binary interference-label (sign of Lumi_weight).

Saves a sorted CSV file: feature, mutual_info
"""

import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

OUTPUT_CSV = "../results/mutual_information_base.csv"
INPUT_H5   = "../data/s2286706/new_Input_CP_Studies_llqq_LinearTerm_20th_October2025.h5"


def main():
    print("Loading dataset...")
    with h5py.File(INPUT_H5, "r") as f:
        df = pd.DataFrame(f["LargeRJet"]["1d"][:])

    # -------------------------------------------
    # 1. Prepare labels (signal = Lumi_weight > 0)
    # -------------------------------------------
    y = df["Lumi_weight"].copy()
    y[y > 0] = 1
    y[y < 0] = 0
    y = y.values.astype(int)

    # -------------------------------------------
    # 2. Keep only feature columns
    # -------------------------------------------
    drop_cols = ["Lumi_weight", "EventNumber", "FJ_flavour", "Type"]
    feature_df = df.drop(columns=drop_cols)

    feature_names = feature_df.columns.tolist()
    X = feature_df.values

    print(f"Number of features: {len(feature_names)}")

    # -------------------------------------------
    # 3. Scale features (MI performs better with scaling)
    # -------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------------------
    # 4. Compute Mutual Information
    # -------------------------------------------
    print("Computing mutual information...")
    mi = mutual_info_classif(
        X_scaled,
        y,
        discrete_features="auto",
        random_state=42
    )

    # -------------------------------------------
    # 5. Save results into a nice ordered CSV
    # -------------------------------------------
    mi_df = pd.DataFrame({
        "Feature": feature_names,
        "MutualInfo": mi
    })

    mi_df = mi_df.sort_values(by="MutualInfo", ascending=False)

    mi_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved MI results → {OUTPUT_CSV}")

    # Print top 10 in console
    print("\nTop 10 features by MI:")
    print(mi_df.head(10))


if __name__ == "__main__":
    main()
