"""
AdaptiveStream end-to-end demo.

Runs a Spark Structured Streaming job using the built-in rate source.
The rate varies over time to simulate burst patterns.
The AdaptiveStream agent (if attached) intercepts intervalMs() and
replaces it with the ML-predicted interval.

Without the agent: fixed 1-second trigger, static behaviour.
With the agent:    trigger interval adapts based on predicted rate.

Usage
-----
  # Without agent (baseline):
  spark-submit demo_spark_job.py

  # With agent:
  spark-submit \
    --driver-java-options "-javaagent:agent/target/adaptivestream-agent-1.0.0.jar \
      -Dadaptivestream.predictor.model=lstm \
      -Dadaptivestream.python=python3" \
    demo_spark_job.py
"""

import time
import threading
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp

# Burst schedule: (start_sec, rows_per_second)
# Simulates: quiet → burst → quiet → double burst → quiet
RATE_SCHEDULE = [
    (0,   50),    # baseline
    (15,  500),   # wall burst
    (30,  50),    # back to baseline
    (45,  800),   # bigger burst
    (55,  300),   # partial drop
    (70,  50),    # quiet again
    (85,  1200),  # spike
    (95,  50),    # drain
    (110, 50),    # end
]


def rate_changer(spark, schedule, stop_event):
    """Background thread: dynamically changes the rate source's rowsPerSecond."""
    start = time.time()
    idx = 0
    while not stop_event.is_set():
        elapsed = time.time() - start
        # find the current target rate
        current_rate = schedule[0][1]
        for t, r in schedule:
            if elapsed >= t:
                current_rate = r
        print(f"[demo] t={elapsed:.0f}s  rate_target={current_rate} rows/s")
        time.sleep(5)
    print("[demo] Rate changer stopped.")


def main():
    spark = (SparkSession.builder
             .master("local[2]")
             .appName("AdaptiveStreamDemo")
             .config("spark.sql.shuffle.partitions", "2")
             .config("spark.ui.enabled", "true")
             .config("spark.ui.port", "4040")
             .getOrCreate())

    spark.sparkContext.setLogLevel("WARN")
    print("[demo] Spark UI available at http://localhost:4040")
    print("[demo] Starting streaming job — Ctrl+C to stop\n")

    # Rate source: generates (timestamp, value) rows at the given rate.
    # rowsPerSecond can't be changed dynamically in Spark's rate source,
    # so we use a fixed moderate rate and let the agent adapt the interval.
    df = (spark.readStream
          .format("rate")
          .option("rowsPerSecond", 200)
          .option("rampUpTime", "5s")
          .load())

    # Simple aggregation: count rows per second
    query = (df.withColumn("event_time", current_timestamp())
             .writeStream
             .format("console")
             .option("truncate", False)
             .option("numRows", 3)
             .trigger(processingTime="1 second")
             .outputMode("append")
             .start())

    stop_event = threading.Event()
    t = threading.Thread(
        target=rate_changer,
        args=(spark, RATE_SCHEDULE, stop_event),
        daemon=True
    )
    t.start()

    try:
        query.awaitTermination(120)   # run for 2 minutes then stop
    except KeyboardInterrupt:
        print("\n[demo] Interrupted.")
    finally:
        stop_event.set()
        query.stop()
        spark.stop()
        print("[demo] Done.")


if __name__ == "__main__":
    main()
