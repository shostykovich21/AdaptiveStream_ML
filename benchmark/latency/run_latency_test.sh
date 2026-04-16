#!/bin/bash
# Run full latency benchmark: generate burst traffic, then measure all approaches
set -e

TOPIC="adaptive-latency-test"
DURATION=60
SPARK_HOME=${SPARK_HOME:-/opt/spark}
KAFKA_HOME=${KAFKA_HOME:-/opt/kafka}
AGENT_JAR="../agent/target/adaptivestream-agent-1.0.0.jar"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Clean previous results
rm -rf /tmp/latency_results /tmp/latency_ckpt

# Recreate topic
$KAFKA_HOME/bin/kafka-topics.sh --delete --topic $TOPIC --bootstrap-server localhost:9092 2>/dev/null || true
sleep 2
$KAFKA_HOME/bin/kafka-topics.sh --create --topic $TOPIC --bootstrap-server localhost:9092 --partitions 2 --replication-factor 1

echo "=== Starting burst generator in background ==="
python3 "$REPO_DIR/generator/burst_generator.py" \
    --topic $TOPIC --duration $DURATION --baseline 100 --burst-mult 8 &
GEN_PID=$!
sleep 3

echo ""
echo "=== Running fixed 2s baseline ==="
$SPARK_HOME/bin/spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.4 \
    "$SCRIPT_DIR/latency_benchmark.py" fixed_2s $DURATION $TOPIC 2>&1 | grep -E "\[Latency|rate_target|events"

echo ""
echo "=== Running fixed 500ms baseline ==="
# Restart generator
wait $GEN_PID 2>/dev/null
python3 "$REPO_DIR/generator/burst_generator.py" \
    --topic $TOPIC --duration $DURATION --baseline 100 --burst-mult 8 &
GEN_PID=$!
sleep 3

$SPARK_HOME/bin/spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.4 \
    "$SCRIPT_DIR/latency_benchmark.py" fixed_500ms $DURATION $TOPIC 2>&1 | grep -E "\[Latency|rate_target|events"

echo ""
echo "=== Running adaptive (with agent) ==="
wait $GEN_PID 2>/dev/null
python3 "$REPO_DIR/generator/burst_generator.py" \
    --topic $TOPIC --duration $DURATION --baseline 100 --burst-mult 8 &
GEN_PID=$!
sleep 3

$SPARK_HOME/bin/spark-submit \
    --conf "spark.driver.extraJavaOptions=-javaagent:$AGENT_JAR" \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.4 \
    "$SCRIPT_DIR/latency_benchmark.py" adaptive $DURATION $TOPIC 2>&1 | grep -E "\[Latency|rate_target|events|\[Adaptive"

wait $GEN_PID 2>/dev/null

echo ""
echo "=== Results ==="
for dir in /tmp/latency_results/*/; do
    mode=$(basename $dir)
    count=$(find $dir -name "*.json" | xargs cat 2>/dev/null | wc -l)
    echo "$mode: $count events recorded"
done
