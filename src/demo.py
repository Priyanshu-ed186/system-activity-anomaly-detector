"""Reproducible Isolation Forest demo using explicitly simulated data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FEATURES = ["cpu_percent", "memory_percent"]


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    baseline = pd.DataFrame({
        "cpu_percent": np.clip(rng.normal(24, 8, 800), 1, 65),
        "memory_percent": np.clip(rng.normal(48, 9, 800), 15, 78),
    })
    ordinary = pd.DataFrame({
        "cpu_percent": np.clip(rng.normal(25, 9, 240), 1, 68),
        "memory_percent": np.clip(rng.normal(49, 10, 240), 15, 80),
    })
    unusual = pd.DataFrame({
        "cpu_percent": np.clip(rng.normal(88, 5, 30), 70, 100),
        "memory_percent": np.clip(rng.normal(86, 6, 30), 68, 100),
    })
    test = pd.concat([ordinary, unusual], ignore_index=True)
    labels = np.r_[np.zeros(len(ordinary), dtype=int),
                   np.ones(len(unusual), dtype=int)]
    order = rng.permutation(len(test))
    return baseline, test.iloc[order].reset_index(drop=True), labels[order]


def main():
    RESULTS.mkdir(exist_ok=True)
    train, test, actual = make_data()
    model = IsolationForest(contamination=0.11, random_state=42)
    model.fit(train[FEATURES])
    # IsolationForest returns 1 for inliers and -1 for outliers.
    predicted = (model.predict(test[FEATURES]) == -1).astype(int)
    test["actual_unusual"] = actual
    test["predicted_unusual"] = predicted
    test.to_csv(RESULTS / "demo_predictions.csv", index=False)

    print("DEMO DATA ONLY: controlled simulated CPU/memory readings")
    print(f"Training rows: {len(train)} normal simulated readings")
    print(f"Test rows: {len(test)} ({int(actual.sum())} injected unusual readings)")
    print(f"Accuracy:          {accuracy_score(actual, predicted):.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(actual, predicted):.3f}")
    print(f"Precision:         {precision_score(actual, predicted, zero_division=0):.3f}")
    print(f"Recall:            {recall_score(actual, predicted, zero_division=0):.3f}")
    print(f"F1 score:          {f1_score(actual, predicted, zero_division=0):.3f}")
    print("Confusion matrix [ [normal correct, normal flagged], [unusual missed, unusual flagged] ]:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))
    print("\nDetailed report:")
    print(classification_report(actual, predicted, target_names=["normal", "unusual"],
                                zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    normal = test["predicted_unusual"] == 0
    ax.scatter(test.loc[normal, "cpu_percent"], test.loc[normal, "memory_percent"],
               s=22, alpha=.65, label="Predicted normal")
    ax.scatter(test.loc[~normal, "cpu_percent"], test.loc[~normal, "memory_percent"],
               s=32, alpha=.8, marker="x", label="Flagged unusual")
    ax.set(title="Simulated CPU and memory anomaly detection",
           xlabel="CPU usage (%)", ylabel="Memory usage (%)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTS / "demo_plot.png", dpi=160)
    print(f"\nSaved: {RESULTS / 'demo_predictions.csv'}")
    print(f"Saved: {RESULTS / 'demo_plot.png'}")


if __name__ == "__main__":
    main()
