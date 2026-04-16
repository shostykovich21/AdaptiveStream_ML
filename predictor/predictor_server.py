"""
AdaptiveStream LSTM Predictor Server
Runs as subprocess, communicates with Java Controller via TCP socket.
Handles: rate history collection, LSTM inference, confidence estimation.
"""

import socket
import json
import time
import threading
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# --- Model ---
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- Rate Collector (background thread) ---
class RateCollector:
    """Collects Kafka consumer rate from Spark metrics or direct observation."""
    def __init__(self, window_size=30):
        self.rates = deque(maxlen=window_size)
        self.window_size = window_size
        self._lock = threading.Lock()

    def add_rate(self, rate):
        with self._lock:
            self.rates.append(rate)

    def get_history(self):
        with self._lock:
            return list(self.rates)

    def is_ready(self):
        return len(self.rates) >= self.window_size

# --- Confidence Estimator ---
def estimate_confidence(model, history, n_forward=5):
    """
    Monte Carlo dropout-style confidence.
    Run multiple forward passes, measure prediction variance.
    High variance = low confidence.
    """
    model.train()  # enable dropout if present
    preds = []
    x = torch.FloatTensor(history).unsqueeze(0).unsqueeze(-1)
    for _ in range(n_forward):
        with torch.no_grad():
            pred = model(x).item()
            preds.append(pred)
    model.eval()

    variance = np.var(preds)
    # Confidence: inverse of normalized variance
    # Low variance = high confidence
    confidence = 1.0 / (1.0 + variance * 100)
    return float(np.clip(confidence, 0.0, 1.0))

# --- Server ---
def main():
    HOST = "127.0.0.1"
    PORT = 9876
    K = 30  # lookback window

    # Load or initialize model
    model = LSTMPredictor()
    model_path = "/home/aayushvbarhate/adaptivestream/models/lstm_predictor.pt"
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"[Predictor] Loaded model from {model_path}")
    except:
        print("[Predictor] No saved model found, using untrained model")
    model.eval()

    # Warmup inference (avoid cold-start latency)
    dummy = torch.randn(1, K, 1)
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    print("[Predictor] Model warmed up")

    # Rate collector
    collector = RateCollector(window_size=K)

    # Simulated rate feed (in production, reads from Spark metrics endpoint)
    def rate_feed():
        """Placeholder: in real system, scrapes Spark's StreamingQueryProgress"""
        import random
        while True:
            # TODO: Replace with actual Spark metrics scraping
            # For now, simulate rates for testing
            rate = 100 + random.gauss(0, 20)
            collector.add_rate(rate)
            time.sleep(1)

    feed_thread = threading.Thread(target=rate_feed, daemon=True)
    feed_thread.start()

    # TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[Predictor] Listening on {HOST}:{PORT}")

    conn, addr = server.accept()
    print(f"[Predictor] Controller connected from {addr}")

    try:
        while True:
            data = conn.recv(1024).decode().strip()
            if not data:
                break

            if data == "predict":
                history = collector.get_history()

                if len(history) < K:
                    # Not enough data yet
                    response = f"predicted_rate:100.0,confidence:0.0"
                else:
                    # Normalize
                    h = np.array(history, dtype=np.float32)
                    mu, std = h.mean(), h.std() + 1e-8
                    normed = (h - mu) / std

                    # Predict
                    x = torch.FloatTensor(normed).unsqueeze(0).unsqueeze(-1)
                    start = time.perf_counter()
                    with torch.no_grad():
                        pred_normed = model(x).item()
                    inference_ms = (time.perf_counter() - start) * 1000

                    # Denormalize
                    predicted_rate = pred_normed * std + mu
                    predicted_rate = max(predicted_rate, 0)

                    # Confidence
                    confidence = estimate_confidence(model, normed)

                    response = f"predicted_rate:{predicted_rate:.2f},confidence:{confidence:.4f}"

                conn.send((response + "\n").encode())

            elif data == "health":
                conn.send(b"ok\n")

            elif data.startswith("rate:"):
                # Manual rate injection (for testing)
                rate = float(data.split(":")[1])
                collector.add_rate(rate)
                conn.send(b"ack\n")

    except Exception as e:
        print(f"[Predictor] Error: {e}")
    finally:
        conn.close()
        server.close()

if __name__ == "__main__":
    main()
