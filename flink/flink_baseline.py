"""
Apache Flink equivalent pipeline for fair comparison with Spark.

Uses PyFlink (Flink's Python API) to consume from the same Kafka topic,
apply the same aggregation, and measure the same latency metrics.

This ensures an apples-to-apples comparison:
  - Same Kafka topic and data
  - Same aggregation logic (count events per window)
  - Same latency measurement (producer_ts → processing time)
  - Same output format (JSON with per-event latency)

Requirements:
  pip install apache-flink==1.18.0
"""

import sys
import json
import time

try:
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.table import StreamTableEnvironment, EnvironmentSettings
    from pyflink.common import WatermarkStrategy, Duration
    FLINK_AVAILABLE = True
except ImportError:
    FLINK_AVAILABLE = False
    print("[Flink] PyFlink not installed. Install with: pip install apache-flink==1.18.0")

TOPIC = sys.argv[1] if len(sys.argv) > 1 else "adaptive-latency-test"
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 60
WINDOW_SIZE = sys.argv[3] if len(sys.argv) > 3 else "2"  # seconds


def run_flink_pipeline():
    if not FLINK_AVAILABLE:
        print("[Flink] Skipping — PyFlink not available")
        return

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)  # match Spark's local[2]

    t_env = StreamTableEnvironment.create(env)

    # Kafka source — same topic as Spark pipeline
    t_env.execute_sql(f"""
        CREATE TABLE kafka_source (
            `ts` DOUBLE,
            `value` INT,
            `rate_target` INT,
            `second` INT,
            `payload` STRING,
            `event_time` AS TO_TIMESTAMP_LTZ(CAST(`ts` * 1000 AS BIGINT), 3),
            `process_time` AS PROCTIME(),
            WATERMARK FOR `event_time` AS `event_time` - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = '{TOPIC}',
            'properties.bootstrap.servers' = 'localhost:9092',
            'properties.group.id' = 'flink-baseline',
            'scan.startup.mode' = 'latest-offset',
            'format' = 'json'
        )
    """)

    # Tumbling window aggregation — equivalent to Spark's window()
    t_env.execute_sql(f"""
        CREATE TABLE results (
            `window_start` TIMESTAMP(3),
            `window_end` TIMESTAMP(3),
            `event_count` BIGINT,
            `avg_rate_target` DOUBLE
        ) WITH (
            'connector' = 'print'
        )
    """)

    t_env.execute_sql(f"""
        INSERT INTO results
        SELECT
            TUMBLE_START(event_time, INTERVAL '{WINDOW_SIZE}' SECOND) AS window_start,
            TUMBLE_END(event_time, INTERVAL '{WINDOW_SIZE}' SECOND) AS window_end,
            COUNT(*) AS event_count,
            AVG(CAST(rate_target AS DOUBLE)) AS avg_rate_target
        FROM kafka_source
        GROUP BY TUMBLE(event_time, INTERVAL '{WINDOW_SIZE}' SECOND)
    """)

    print(f"[Flink] Pipeline running with {WINDOW_SIZE}s tumbling window for {DURATION}s...")
    time.sleep(DURATION)
    print("[Flink] Done")


def run_flink_latency_measurement():
    """
    Per-event latency measurement pipeline for Flink.
    Mirrors the Spark latency_benchmark.py output format.
    """
    if not FLINK_AVAILABLE:
        print("[Flink] Skipping — PyFlink not available")
        return

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)

    t_env = StreamTableEnvironment.create(env)

    t_env.execute_sql(f"""
        CREATE TABLE kafka_source (
            `ts` DOUBLE,
            `value` INT,
            `rate_target` INT,
            `second` INT,
            `payload` STRING,
            `proc_ts` AS PROCTIME()
        ) WITH (
            'connector' = 'kafka',
            'topic' = '{TOPIC}',
            'properties.bootstrap.servers' = 'localhost:9092',
            'properties.group.id' = 'flink-latency',
            'scan.startup.mode' = 'latest-offset',
            'format' = 'json'
        )
    """)

    # Compute per-event latency
    t_env.execute_sql("""
        CREATE TABLE latency_output (
            `producer_ts` DOUBLE,
            `rate_target` INT,
            `second` INT,
            `e2e_latency_ms` DOUBLE
        ) WITH (
            'connector' = 'filesystem',
            'path' = '/tmp/latency_results/flink/',
            'format' = 'json'
        )
    """)

    t_env.execute_sql("""
        INSERT INTO latency_output
        SELECT
            ts AS producer_ts,
            rate_target,
            `second`,
            (UNIX_TIMESTAMP(CAST(proc_ts AS TIMESTAMP)) - ts) * 1000 AS e2e_latency_ms
        FROM kafka_source
    """)

    print(f"[Flink] Latency measurement running for {DURATION}s...")
    time.sleep(DURATION)
    print("[Flink] Results written to /tmp/latency_results/flink/")


if __name__ == "__main__":
    mode = sys.argv[4] if len(sys.argv) > 4 else "aggregate"
    if mode == "latency":
        run_flink_latency_measurement()
    else:
        run_flink_pipeline()
