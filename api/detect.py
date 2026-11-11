"""Vercel function: classify an entered CPU/memory pair against demo data."""

import json
from http.server import BaseHTTPRequestHandler

import numpy as np
from sklearn.ensemble import IsolationForest


def make_demo_model():
    rng = np.random.default_rng(42)
    baseline = np.column_stack((
        np.clip(rng.normal(24, 8, 800), 1, 65),
        np.clip(rng.normal(48, 9, 800), 15, 78),
    ))
    return IsolationForest(contamination=0.05, random_state=42).fit(baseline)


MODEL = make_demo_model()


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send_json(200, {
            "service": "System Activity Anomaly Demo",
            "note": "Predictions use a simulated reference pattern.",
        })

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 4096:
                return self._send_json(400, {"error": "Send a small JSON body with CPU and memory values."})
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            cpu = float(payload["cpu_percent"])
            memory = float(payload["memory_percent"])
            if not (0 <= cpu <= 100 and 0 <= memory <= 100):
                return self._send_json(400, {"error": "CPU and memory must be between 0 and 100."})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return self._send_json(400, {"error": "Provide numeric cpu_percent and memory_percent values."})

        sample = np.array([[cpu, memory]], dtype=float)
        is_outlier = int(MODEL.predict(sample)[0]) == -1
        score = float(MODEL.decision_function(sample)[0])
        self._send_json(200, {
            "cpu_percent": round(cpu, 2),
            "memory_percent": round(memory, 2),
            "is_outlier": is_outlier,
            "label": "unusual" if is_outlier else "within_demo_pattern",
            "score": score,
            "note": "Relative to a simulated reference pattern; not a security diagnosis.",
        })
