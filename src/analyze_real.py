"""Flag relative outliers in a locally collected, unlabeled activity CSV."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "system_activity.csv"
RESULTS = ROOT / "results"
FEATURES = ["cpu_percent", "memory_percent"]


def main():
    if not INPUT.exists():
        raise SystemExit("No collected CSV found. Run src/collect.py first.")
    data = pd.read_csv(INPUT)
    if len(data) < 20:
        raise SystemExit("Collect at least 20 readings before analysis.")
    if data[FEATURES].isna().any().any():
        raise SystemExit("The input CSV contains missing CPU or memory values.")

    RESULTS.mkdir(exist_ok=True)
    model = IsolationForest(contamination=0.05, random_state=42)
    data["is_outlier"] = model.fit_predict(data[FEATURES]) == -1
    data["anomaly_score"] = model.decision_function(data[FEATURES])
    output = RESULTS / "real_activity_analysis.csv"
    data.to_csv(output, index=False)

    count = int(data["is_outlier"].sum())
    print("UNLABELED LOCAL OBSERVATIONS: outliers are relative to this sample only")
    print(f"Readings analyzed: {len(data)}")
    print(f"Relative outliers flagged: {count}")
    print("CPU range: {:.2f}% to {:.2f}%".format(
        data["cpu_percent"].min(), data["cpu_percent"].max()))
    print("Memory range: {:.2f}% to {:.2f}%".format(
        data["memory_percent"].min(), data["memory_percent"].max()))
    print(f"Saved: {output}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    flagged = data["is_outlier"]
    series = [("cpu_percent", "CPU usage (%)"),
              ("memory_percent", "Memory usage (%)")]
    for ax, (column, label) in zip(axes, series):
        ax.plot(data.index, data[column], label=label, linewidth=1.5)
        ax.scatter(data.index[flagged], data.loc[flagged, column],
                   color="crimson", marker="x", s=48,
                   label="Sample flagged by model")
        ax.set_ylabel("Usage (%)")
        ax.legend(loc="best")
        ax.grid(alpha=.2)
    axes[0].set_title("Local readings and relative outliers (separate scales)")
    axes[-1].set_xlabel("Reading number (one per second)")
    fig.tight_layout()
    chart = RESULTS / "real_activity_plot.png"
    fig.savefig(chart, dpi=160)
    print(f"Saved: {chart}")
    print("An outlier is not proof of malware, intrusion, or a security incident.")


if __name__ == "__main__":
    main()
