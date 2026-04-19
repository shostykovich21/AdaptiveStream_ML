"""
AdaptiveStream Predictor Server.
Communicates with Java Controller via TCP.
Collects real Spark streaming rates via REST API.

Usage
-----
    python predictor_server.py                   # default: lstm
    python predictor_server.py --model tcn
    python predictor_server.py --model ens_wtd   # after adding ensemble support
    python predictor_server.py --model lstm --port 9876 --spark-app MyApp
"""

import argparse
import socket
import time
import numpy as np
import torch
from models import MODELS
from metrics_collector import SparkMetricsCollector
from config import K, MODEL_DIR


def estimate_confidence(model, history_tensor, n_passes=10):
    """
    MC Dropout confidence estimation.
    Run multiple forward passes in train mode (enables dropout), measure variance.
    High variance = low confidence. Returns 1.0 for models without dropout layers.
    """
    model.train()
    preds = []
    for _ in range(n_passes):
        with torch.no_grad():
            preds.append(model(history_tensor).item())
    model.eval()
    std = np.std(preds)
    return float(np.clip(1.0 / (1.0 + std * 10), 0.0, 1.0))


def main():
    parser = argparse.ArgumentParser(description="AdaptiveStream predictor server")
    parser.add_argument("--model", default="lstm", choices=list(MODELS.keys()),
                        help=f"model architecture to serve (default: lstm)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--spark-url", default="http://localhost:4040",
                        help="Spark UI base URL")
    parser.add_argument("--spark-app", default=None,
                        help="Spark app name to attach to (default: first available)")
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing model checkpoints "
                             "(default: models/ at project root)")
    args = parser.parse_args()

    from pathlib import Path
    model_dir  = Path(args.model_dir) if args.model_dir else MODEL_DIR
    # Load model from registry — no longer hardwired to LSTM
    model = MODELS[args.model]()
    model_path = model_dir / f"{args.model}_predictor.pt"
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        print(f"[Predictor] Loaded {args.model} from {model_path}")
    except Exception as e:
        print(f"[Predictor] FATAL: could not load {args.model} from {model_path}: {e}")
        print("[Predictor] Refusing to serve with uninitialised weights — exiting.")
        raise SystemExit(1)
    model.eval()

    # Warmup — avoid cold start latency on first real prediction
    dummy = torch.randn(1, K, 1)
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    print(f"[Predictor] {args.model} warmed up (10 dummy inferences)")

    # Start metrics collector
    collector = SparkMetricsCollector(
        spark_ui_url=args.spark_url,
        window_size=K,
        poll_interval=1.0,
        app_name=args.spark_app,
    )
    collector.start()
    print(f"[Predictor] Metrics collector started (polling {args.spark_url})")
    if args.spark_app:
        print(f"[Predictor] Attached to Spark app: {args.spark_app}")

    # TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[Predictor] Listening on {args.host}:{args.port}")

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
                    # window not full yet — return partial mean rather than
                    # a hardcoded synthetic baseline (100.0 was only valid for
                    # the synthetic training distribution, not real workloads)
                    fallback = float(np.mean(history)) if history else 0.0
                    response = f"predicted_rate:{fallback:.2f},confidence:0.0"
                else:
                    h   = np.array(history, dtype=np.float32)
                    mu  = h.mean()
                    std = max(float(h.std()), mu * 0.05) + 1e-8
                    x   = torch.FloatTensor((h - mu) / std).unsqueeze(0).unsqueeze(-1)

                    with torch.no_grad():
                        pred_normed = model(x).item()

                    predicted_rate = max(pred_normed * std + mu, 0.0)
                    confidence     = estimate_confidence(model, x)
                    response       = f"predicted_rate:{predicted_rate:.2f},confidence:{confidence:.4f}"

                conn.send((response + "\n").encode())

            elif data == "health":
                ready = "ready" if collector.is_ready() else "warming_up"
                conn.send(
                    f"ok,status:{ready},history_len:{len(collector.get_history())}\n".encode()
                )

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
