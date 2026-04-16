# Flink Baseline

Equivalent streaming pipeline in Apache Flink for comparison with AdaptiveStream.

## Setup

```bash
pip install apache-flink==1.18.0
```

## Usage

```bash
# Same aggregation as Spark baseline, 2s tumbling window
python3 flink_baseline.py adaptive-latency-test 60 2

# Per-event latency measurement (for benchmark comparison)
python3 flink_baseline.py adaptive-latency-test 60 2 latency
```

## Fairness guarantees

To ensure a fair comparison with Spark + AdaptiveStream:

1. Same Kafka topic, same data, same producer
2. Same parallelism (2 threads)
3. Same aggregation (count per window)
4. Same latency metric (producer timestamp → processing timestamp)
5. Same output format (JSON with per-event latency)
6. Both use event-time windowing with 5s watermark tolerance
