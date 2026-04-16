"""
AdaptiveStream LSTM Predictor Server.
Communicates with Java Controller via TCP.
Collects real Spark streaming rates via REST API.
"""

import socket
import time
import numpy as np
import torch
from pathlib import Path
from models import LSTMPredictor
from metrics_collector import SparkMetricsCollector, DriverSideCollector


def estimate_confidence(model, history_tensor, n_passes=10):
    """
    MC Dropout confidence estimation.
    Model has dropout layers — run multiple forward passes in train mode,
    measure variance. High variance = low confidence.
    """
    model.train()  # enables dropout
    preds = []
    for _ in range(n_passes):
        with torch.no_grad():
            p = model(history_tensor).item()
            preds.append(p)
    model.eval()

    std = np.std(preds)
    # Confidence inversely proportional to prediction spread
    # Normalized so baseline noise gives ~0.8-0.9 confidence
    confidence = 1.0 / (1.0 + std * 10)
    return float(np.clip(confidence, 0.0, 1.0))


def main():
    HOST = "127.0.0.1"
    PORT = 9876
    K = 30

    # Load model
    model = LSTMPredictor()
    model_path = Path(__file__).parent.parent / "models" / "lstm_predictor.pt"
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        print(f"[Predictor] Loaded model from {model_path}")
    except Exception as e:
        print(f"[Predictor] No saved model ({e}), using untrained")
    model.eval()

    # Warmup — avoid cold start latency on first real prediction
    dummy = torch.randn(1, K, 1)
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    print("[Predictor] Model warmed up (10 dummy inferences)")

    # Start metrics collector (polls Spark REST API)
    collector = SparkMetricsCollector(
        spark_ui_url="http://localhost:4040",
        window_size=K,
        poll_interval=1.0
    )
    collector.start()
    print("[Predictor] Metrics collector started (polling Spark :4040)")

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
                    response = f"predicted_rate:100.0,confidence:0.0"
                else:
                    h = np.array(history, dtype=np.float32)
                    mu, std = h.mean(), h.std() + 1e-8
                    normed = (h - mu) / std

                    x = torch.FloatTensor(normed).unsqueeze(0).unsqueeze(-1)
                    with torch.no_grad():
                        pred_normed = model(x).item()

                    predicted_rate = max(pred_normed * std + mu, 0)
                    confidence = estimate_confidence(model, x)

                    response = f"predicted_rate:{predicted_rate:.2f},confidence:{confidence:.4f}"

                conn.send((response + "\n").encode())

            elif data == "health":
                ready = "ready" if collector.is_ready() else "warming_up"
                conn.send(f"ok,status:{ready},history_len:{len(collector.get_history())}\n".encode())

            elif data.startswith("rate:"):
                # Manual rate injection (testing)
                rate = float(data.split(":")[1])
                collector.add_rate_manual(rate)
                conn.send(b"ack\n")

    except Exception as e:
        print(f"[Predictor] Error: {e}")
    finally:
        collector.stop()
        conn.close()
        server.close()

if __name__ == "__main__":
    main()
