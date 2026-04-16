"""
Benchmark framework — runs all approaches on same workload, collects metrics.
"""
import subprocess
import time
import json
import csv
import os

APPROACHES = {
    "fixed_2s": {
        "cmd": ["spark-submit", "../baselines/fixed_baseline.py", "2 seconds", "60"],
        "type": "baseline"
    },
    "fixed_500ms": {
        "cmd": ["spark-submit", "../baselines/fixed_baseline.py", "500 milliseconds", "60"],
        "type": "baseline"
    },
    "reactive": {
        "cmd": ["spark-submit", "../baselines/reactive_baseline.py"],
        "type": "baseline"
    },
    "adaptive": {
        "cmd": ["spark-submit",
                "--conf", "spark.driver.extraJavaOptions=-javaagent:../agent/target/adaptivestream-agent-1.0.0.jar",
                "../baselines/fixed_baseline.py", "2 seconds", "60"],
        "type": "adaptive"
    },
}

def run_approach(name, config):
    print(f"\n{'='*50}")
    print(f"Running: {name}")
    print(f"{'='*50}")

    start = time.time()
    result = subprocess.run(config["cmd"], capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start

    return {
        "approach": name,
        "elapsed_sec": round(elapsed, 2),
        "exit_code": result.returncode,
        "stdout_lines": len(result.stdout.split("\n")),
    }

if __name__ == "__main__":
    results = []
    for name, config in APPROACHES.items():
        try:
            r = run_approach(name, config)
            results.append(r)
            print(f"  -> {name}: {r['elapsed_sec']}s, exit={r['exit_code']}")
        except Exception as e:
            print(f"  -> {name}: FAILED - {e}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["approach", "elapsed_sec", "exit_code", "stdout_lines"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to results/benchmark_results.csv")
