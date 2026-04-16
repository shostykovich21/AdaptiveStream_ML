# AdaptiveStream

A drop-in JVM agent that enables proactive ML-based adaptive windowing for Apache Spark Structured Streaming — without source modification, pipeline restart, or state loss.

## Problem

Spark Structured Streaming uses fixed window intervals set at deployment time. Real-world streams are bursty. A window sized for 100 msg/s will spike latency during bursts and waste compute during quiet periods.

## Solution

AdaptiveStream instruments Spark's `ProcessingTimeExecutor` at the bytecode level using ByteBuddy. An LSTM model predicts incoming message rates and dynamically adjusts the trigger interval — before the burst hits.

```bash
# One flag. Existing job unchanged.
spark-submit \
  --conf spark.driver.extraJavaOptions=-javaagent:adaptivestream-agent.jar \
  your_streaming_job.py
```

## Architecture

```
Kafka → Spark Structured Streaming ← dynamic interval ← Controller ← LSTM Predictor
                (unmodified)              (AtomicLong)     (Java)      (Python/PyTorch)
```

## Project Structure

```
agent/          JVM bytecode agent (Java, ByteBuddy)
predictor/      LSTM rate predictor server + training pipeline
generator/      Synthetic burst traffic generator
baselines/      Fixed and reactive window baselines
benchmark/      Benchmark runner and metrics collection
dashboard/      Live monitoring dashboard
models/         Trained model checkpoints (git-ignored, reproducible)
```

## Quick Start

### Prerequisites
- Java 8+, Maven 3.x
- Python 3.8+, PyTorch, PySpark
- Apache Kafka 3.x
- Apache Spark 3.4+

### Build

```bash
# Build agent
cd agent && mvn package

# Train LSTM
python3 predictor/train.py

# Generate burst data into Kafka
python3 generator/burst_generator.py --topic adaptive-stream --duration 300

# Run with adaptive windowing
spark-submit \
  --conf spark.driver.extraJavaOptions=-javaagent:agent/target/adaptivestream-agent-1.0.0.jar \
  baselines/fixed_baseline.py
```

### Run benchmarks

```bash
cd benchmark && python3 run_benchmark.py
```

## Research Questions

1. Can a JVM bytecode agent instrument Spark's trigger execution to enable dynamic window sizing without pipeline restart or state loss?
2. Does LSTM-based proactive window prediction reduce peak end-to-end latency compared to reactive and fixed baselines under bursty workloads?
3. How does AdaptiveStream compare to Apache Flink's native adaptive streaming?

## License

MIT
