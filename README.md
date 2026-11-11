# System Activity Anomaly Detector

A small Python project that demonstrates anomaly detection in computer resource activity. The demo trains an Isolation Forest on simulated normal CPU and memory readings, then checks whether injected unusual readings are flagged. The simulator is clearly labeled; its results are not measurements of real computer attacks.

## Vercel web demo

The web page lets a visitor enter CPU and memory values and calls a Python Vercel Function. It uses a generated reference pattern to demonstrate the model; it does not monitor the visitor's computer.

To deploy, import this GitHub repository as a new project in Vercel and use the default settings. Vercel serves `index.html` and the Python function at `api/detect.py`.

## Setup in Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the reproducible demo

```powershell
python src/demo.py
```

The demo prints evaluation metrics and writes a CSV and chart in `results/`.

## Collect readings from this computer

```powershell
python src/collect.py --seconds 120 --output data/system_activity.csv
```

This records ordinary resource usage for exploration. It does not label activity as malicious. Keep the computer in normal use while collecting a baseline. Avoid collecting private process names or other personal data.

Analyze the collected observations for relative outliers:

```powershell
python src/analyze_real.py
```

This fits an Isolation Forest to the collected sample and flags about five percent as relative outliers. Since the readings are unlabeled and the model is fitted on this short sample, this is exploratory analysis, not a validated detector or security alert.

## Project limits

An anomaly means a reading differs from the learned pattern; it does not prove a virus, intrusion, or other security incident. The included demo uses controlled synthetic data so its evaluation can be repeated. Real-world performance would need labeled observations and broader testing.
