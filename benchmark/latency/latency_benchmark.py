"""
Measures actual end-to-end latency: event_timestamp (Kafka produce time) → Spark output time.
Runs all approaches (fixed, reactive, adaptive) on the same Kafka burst workload.
Outputs per-event latency CSVs for analysis.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, current_timestamp, expr,
    unix_timestamp, lit, count, avg, percentile_approx
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import time
import sys
import os

MODE = sys.argv[1] if len(sys.argv) > 1 else "fixed_2s"
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 60
TOPIC = sys.argv[3] if len(sys.argv) > 3 else "adaptive-latency-test"
OUTPUT_DIR = f"/tmp/latency_results/{MODE}"

spark = SparkSession.builder \
    .appName(f"LatencyBench-{MODE}") \
    .master("local[2]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Read from Kafka
raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", TOPIC) \
    .option("startingOffsets", "latest") \
    .load()

schema = StructType([
    StructField("ts", DoubleType()),          # producer timestamp (epoch seconds)
    StructField("value", IntegerType()),
    StructField("rate_target", IntegerType()),
    StructField("second", IntegerType()),
    StructField("payload", StringType())
])

parsed = raw.select(
    from_json(col("value").cast("string"), schema).alias("data"),
    col("timestamp").alias("kafka_ingest_ts")  # Kafka broker timestamp
).select(
    col("data.ts").alias("producer_ts"),
    col("data.value"),
    col("data.rate_target"),
    col("data.second"),
    col("kafka_ingest_ts")
)

# Add processing timestamp and compute latency
with_latency = parsed.withColumn(
    "process_ts", current_timestamp()
).withColumn(
    "e2e_latency_ms",
    (unix_timestamp("process_ts").cast("double") - col("producer_ts")) * 1000
).withColumn(
    "kafka_latency_ms",
    (unix_timestamp("process_ts").cast("double") - unix_timestamp("kafka_ingest_ts").cast("double")) * 1000
)

# Determine trigger interval based on mode
intervals = {
    "fixed_500ms": "500 milliseconds",
    "fixed_2s": "2 seconds",
    "fixed_5s": "5 seconds",
    "adaptive": "2 seconds",  # agent overrides this dynamically
}
trigger_interval = intervals.get(MODE, "2 seconds")

# Write latency data to files
query = with_latency \
    .select("producer_ts", "rate_target", "second", "e2e_latency_ms", "kafka_latency_ms") \
    .writeStream \
    .format("json") \
    .option("path", OUTPUT_DIR) \
    .option("checkpointLocation", f"/tmp/latency_ckpt/{MODE}") \
    .trigger(processingTime=trigger_interval) \
    .outputMode("append") \
    .start()

# Also print rolling stats
stats_query = with_latency \
    .groupBy("rate_target") \
    .agg(
        count("*").alias("events"),
        avg("e2e_latency_ms").alias("avg_latency_ms"),
        percentile_approx("e2e_latency_ms", 0.5).alias("p50_ms"),
        percentile_approx("e2e_latency_ms", 0.95).alias("p95_ms"),
        percentile_approx("e2e_latency_ms", 0.99).alias("p99_ms"),
    ) \
    .writeStream \
    .format("console") \
    .outputMode("complete") \
    .trigger(processingTime="10 seconds") \
    .start()

print(f"[LatencyBench] Mode={MODE}, trigger={trigger_interval}, running for {DURATION}s...")
time.sleep(DURATION)

query.stop()
stats_query.stop()
spark.stop()

print(f"[LatencyBench] Results written to {OUTPUT_DIR}")
