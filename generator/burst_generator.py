"""
Synthetic burst traffic generator for training and testing.
Produces Kafka messages with controllable burst patterns.
"""

import json
import time
import random
import argparse
import subprocess
import numpy as np

def generate_burst_pattern(duration_sec=300, baseline_rate=100, burst_mult=8,
                           burst_prob=0.1, burst_duration=30, noise_std=10):
    """Generate a rate schedule (rates per second for each second)."""
    rates = []
    i = 0
    while i < duration_sec:
        if random.random() < burst_prob and i + burst_duration < duration_sec:
            # Burst: ramp up -> hold -> ramp down
            third = burst_duration // 3
            ramp_up = np.linspace(baseline_rate, baseline_rate * burst_mult, third)
            hold = np.full(third, baseline_rate * burst_mult)
            ramp_down = np.linspace(baseline_rate * burst_mult, baseline_rate, third)
            burst = np.concatenate([ramp_up, hold, ramp_down])
            rates.extend(burst + np.random.normal(0, noise_std, len(burst)))
            i += len(burst)
        else:
            rates.append(baseline_rate + np.random.normal(0, noise_std))
            i += 1
    return [max(1, int(r)) for r in rates[:duration_sec]]

def produce_to_kafka(topic, broker, rates, payload_size=100):
    """Send messages to Kafka following the rate schedule."""
    proc = subprocess.Popen(
        ["kafka-console-producer.sh", "--broker-list", broker, "--topic", topic],
        stdin=subprocess.PIPE, text=True
    )

    total_sent = 0
    for sec, rate in enumerate(rates):
        interval = 1.0 / max(rate, 1)
        for _ in range(rate):
            msg = json.dumps({
                "ts": time.time(),
                "value": random.randint(1, 1000),
                "rate_target": rate,
                "second": sec,
                "payload": "x" * payload_size
            })
            proc.stdin.write(msg + "\n")
            proc.stdin.flush()
            time.sleep(interval)
        total_sent += rate

        if (sec + 1) % 30 == 0:
            print(f"[Generator] {sec+1}s elapsed, {total_sent} msgs sent, current rate: {rate}/s")

    proc.stdin.close()
    proc.wait()
    print(f"[Generator] Done. Total: {total_sent} messages in {len(rates)} seconds.")
    return total_sent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="adaptive-stream")
    parser.add_argument("--broker", default="localhost:9092")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--baseline", type=int, default=100)
    parser.add_argument("--burst-mult", type=int, default=8)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    args = parser.parse_args()

    print(f"[Generator] Generating {args.duration}s burst pattern (baseline={args.baseline}, burst={args.burst_mult}x)")
    rates = generate_burst_pattern(
        duration_sec=args.duration,
        baseline_rate=args.baseline,
        burst_mult=args.burst_mult,
        burst_prob=args.burst_prob,
    )
    print(f"[Generator] Rate schedule: min={min(rates)}, max={max(rates)}, mean={np.mean(rates):.0f}")
    produce_to_kafka(args.topic, args.broker, rates)
