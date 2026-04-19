"""
run.py — AdaptiveStream ML Pipeline Orchestrator

Automates the full pipeline:
  Step 1: train.py          — train all 9 models
  Step 2: evaluate_stream.py  — synthetic holdout evaluation
  Step 3: evaluate_stream2.py — real Spark job evaluation

Each step's output is logged to logs/run_{timestamp}/ (or the iteration
folder if --iteration is supplied) and streamed to the console.
Any step failure stops the pipeline and prints a clear diagnostic message.

Usage:
  python run.py                           # full pipeline
  python run.py --skip-train              # skip training (models already exist)
  python run.py --skip-eval1              # skip synthetic eval
  python run.py --skip-eval2              # skip real Spark eval
  python run.py --eval2-duration 300      # longer real eval (300s)
  python run.py --lag                     # use Kafka-lag feature (iteration 2)
  python run.py --iteration 2 --iteration-name kafka_lag   # store in iterations/
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PREDICTOR_DIR  = Path(__file__).parent / "predictor"
LOG_ROOT       = Path(__file__).parent / "logs"
ITERATIONS_DIR = Path(__file__).parent / "iterations"


def ts():
    return datetime.now().strftime("%H:%M:%S")


def banner(msg):
    width = 68
    print(f"\n{'─' * width}")
    print(f"  {msg}")
    print(f"{'─' * width}")


def run_step(name, cmd, log_path, env=None):
    """
    Run a subprocess step, tee output to both console and log file.
    Returns (success: bool, elapsed: float).
    """
    banner(f"[{ts()}] {name}")
    print(f"  Command : {' '.join(cmd)}")
    print(f"  Log     : {log_path}")
    print()

    start = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
        proc.wait()

    elapsed = time.time() - start
    success = proc.returncode == 0

    status = "PASSED" if success else f"FAILED (exit {proc.returncode})"
    print(f"\n  [{ts()}] {name} → {status}  ({elapsed:.0f}s)\n")

    if not success:
        print(f"  ✗ Check log for details: {log_path}")
        print()
        print("  Common causes:")
        if "train" in name.lower():
            print("  - No GPU/CUDA available (CPU training is fine, just slower)")
            print("  - Missing dependency: pip install torch numpy")
        elif "eval2" in name.lower():
            print("  - Spark not starting: check port 4040 is free")
            print("  - Socket server not connecting: check firewall/port 9999")
            print("  - No models found: run train.py first (or use --skip-eval2)")
        elif "eval1" in name.lower():
            print("  - No model checkpoints in models/ directory")
            print("  - Run train.py first or remove --skip-train")
        print()

    return success, elapsed


def print_summary(results):
    banner("Pipeline Summary")
    total_elapsed = sum(e for _, e in results.values())
    all_passed    = all(s for s, _ in results.values())

    for step, (success, elapsed) in results.items():
        icon   = "✓" if success else "✗"
        status = "passed" if success else "FAILED"
        print(f"  {icon}  {step:<30} {status}  ({elapsed:.0f}s)")

    print(f"\n  Total time: {total_elapsed:.0f}s")
    if all_passed:
        print("  All steps passed.\n")
    else:
        failed = [s for s, (ok, _) in results.items() if not ok]
        print(f"  Failed steps: {', '.join(failed)}")
        print("  Check logs above for details.\n")


def main():
    parser = argparse.ArgumentParser(description="AdaptiveStream ML pipeline")
    parser.add_argument("--skip-train",  action="store_true")
    parser.add_argument("--skip-eval1",  action="store_true")
    parser.add_argument("--skip-eval2",  action="store_true")
    parser.add_argument("--eval2-duration", type=int, default=120,
                        help="Duration for real Spark evaluation in seconds")
    parser.add_argument("--eval2-mode", choices=["socket", "kafka"],
                        default="socket")
    parser.add_argument("--kafka-broker", default="localhost:9092")
    parser.add_argument("--log-dir", default=None,
                        help="Override log directory (defaults to logs/run_TS or iterations/)")
    parser.add_argument("--lag", action="store_true",
                        help="Train and evaluate with Kafka lag as 2nd input feature")
    parser.add_argument("--log-uniform", action="store_true",
                        help="Use log-uniform baseline dataset (data2.py)")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Iteration number (stores logs in iterations/iteration_N_name/)")
    parser.add_argument("--iteration-name", default="unnamed",
                        help="Short name for this iteration (e.g. kafka_lag)")
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.log_dir:
        log_dir = Path(args.log_dir)
    elif args.iteration is not None:
        iter_folder = f"iteration_{args.iteration}_{args.iteration_name}"
        log_dir = ITERATIONS_DIR / iter_folder
    else:
        log_dir = LOG_ROOT / f"run_{run_ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    banner(f"AdaptiveStream ML Pipeline  —  run_{run_ts}")
    print(f"  Log directory : {log_dir}")
    print(f"  Skip train    : {args.skip_train}")
    print(f"  Skip eval1    : {args.skip_eval1}")
    print(f"  Skip eval2    : {args.skip_eval2}")
    print(f"  Eval2 mode    : {args.eval2_mode}  ({args.eval2_duration}s)")
    print(f"  Lag feature   : {args.lag}")
    print(f"  Log-uniform   : {args.log_uniform}")
    if args.iteration is not None:
        print(f"  Iteration     : {args.iteration} ({args.iteration_name})")

    python  = sys.executable
    results = {}

    # ── Step 1: Train ──────────────────────────────────────────────────────────
    if not args.skip_train:
        train_cmd = [python, str(PREDICTOR_DIR / "train.py")]
        if args.lag:
            train_cmd.append("--lag")
        if args.log_uniform:
            train_cmd.append("--log-uniform")
        ok, elapsed = run_step(
            name     = "Step 1: Training (all 9 models)",
            cmd      = train_cmd,
            log_path = log_dir / "train.log",
        )
        results["train"] = (ok, elapsed)
        if not ok:
            print("  Pipeline stopped after training failure.")
            print_summary(results)
            sys.exit(1)
    else:
        print(f"\n  [skip] Step 1: Training (--skip-train)")

    # ── Step 2: Synthetic holdout eval ────────────────────────────────────────
    if not args.skip_eval1:
        eval1_cmd = [python, str(PREDICTOR_DIR / "evaluate_stream.py")]
        if args.lag:
            eval1_cmd.append("--lag")
        if args.log_uniform:
            eval1_cmd.append("--log-uniform")
        ok, elapsed = run_step(
            name     = "Step 2: evaluate_stream.py (synthetic holdout)",
            cmd      = eval1_cmd,
            log_path = log_dir / "eval_synthetic.log",
        )
        results["eval_synthetic"] = (ok, elapsed)
        if not ok:
            print("  Continuing to real eval despite eval1 failure.")
    else:
        print(f"\n  [skip] Step 2: evaluate_stream.py (--skip-eval1)")

    # ── Step 3: Real Spark eval ───────────────────────────────────────────────
    if not args.skip_eval2:
        eval2_cmd = [
            python,
            str(PREDICTOR_DIR / "evaluate_stream2.py"),
            "--duration",  str(args.eval2_duration),
            "--mode",      args.eval2_mode,
            "--log-dir",   str(log_dir),
        ]
        if args.eval2_mode == "kafka":
            eval2_cmd += ["--kafka-broker", args.kafka_broker]
        if args.lag:
            eval2_cmd.append("--lag")

        ok, elapsed = run_step(
            name     = "Step 3: evaluate_stream2.py (real Spark job)",
            cmd      = eval2_cmd,
            log_path = log_dir / "eval_real.log",
        )
        results["eval_real"] = (ok, elapsed)
    else:
        print(f"\n  [skip] Step 3: evaluate_stream2.py (--skip-eval2)")

    print_summary(results)

    any_failed = any(not ok for ok, _ in results.values())
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
