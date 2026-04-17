"""
Smoke test for predictor_server.py — no Spark or Java required.

Starts the server as a subprocess, drives the TCP protocol, checks
that every response parses correctly, then shuts down cleanly.

Usage
-----
    python test_server.py                 # tests lstm (must have lstm_predictor.pt)
    python test_server.py --model tcn
    python test_server.py --inject        # inject synthetic rates instead of Spark poll
"""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

PREDICTOR = Path(__file__).parent / "predictor_server.py"
HOST, PORT = "127.0.0.1", 9876


def send(sock, msg):
    sock.sendall((msg + "\n").encode())
    return sock.makefile().readline().strip()


def run_test(model, inject_rates):
    print(f"\n[test] Starting server: model={model}")
    proc = subprocess.Popen(
        [sys.executable, str(PREDICTOR), "--model", model],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Wait for server to be ready
    for _ in range(15):
        time.sleep(0.5)
        try:
            sock = socket.create_connection((HOST, PORT), timeout=1)
            break
        except OSError:
            pass
    else:
        out, _ = proc.communicate(timeout=3)
        print("[test] FAIL — server did not start in time")
        print(out)
        proc.kill()
        return False

    try:
        f = sock.makefile()

        def send_cmd(cmd):
            sock.sendall((cmd + "\n").encode())
            return f.readline().strip()

        # ── health check ──────────────────────────────────────────────────────
        resp = send_cmd("health")
        print(f"[test] health  → {resp}")
        assert resp.startswith("ok"), f"unexpected health response: {resp}"
        assert "status:" in resp

        # ── inject synthetic rates (bypasses Spark dependency) ────────────────
        if inject_rates:
            print("[test] injecting 30 synthetic rate values...")
            for rate in [100 + i * 5 for i in range(30)]:
                r = send_cmd(f"rate:{rate}")
                assert r == "ack", f"unexpected ack: {r}"

            # give collector a moment to register them
            time.sleep(0.2)

            resp = send_cmd("health")
            print(f"[test] health (after inject) → {resp}")
            assert "status:ready" in resp, f"expected ready, got: {resp}"

        # ── predict ───────────────────────────────────────────────────────────
        resp = send_cmd("predict")
        print(f"[test] predict → {resp}")
        assert "predicted_rate:" in resp, f"missing predicted_rate: {resp}"
        assert "confidence:" in resp, f"missing confidence: {resp}"

        rate_val = float(resp.split("predicted_rate:")[1].split(",")[0])
        conf_val = float(resp.split("confidence:")[1])
        print(f"[test]   predicted_rate={rate_val:.2f}  confidence={conf_val:.4f}")
        assert rate_val >= 0, "negative rate"
        assert 0.0 <= conf_val <= 1.0, "confidence out of range"

        if inject_rates:
            assert conf_val > 0, "confidence should be >0 with full window"

        print(f"\n[test] PASS — {model} server responds correctly to all commands")
        return True

    except AssertionError as e:
        print(f"[test] FAIL — {e}")
        return False
    except Exception as e:
        print(f"[test] ERROR — {e}")
        return False
    finally:
        sock.close()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        # drain output
        for line in proc.stdout:
            print(f"  [server] {line}", end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm")
    parser.add_argument("--inject", action="store_true",
                        help="inject 30 synthetic rates to fill the window")
    args = parser.parse_args()

    ok = run_test(args.model, inject_rates=args.inject)
    sys.exit(0 if ok else 1)
