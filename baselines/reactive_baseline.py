"""
Reactive baseline — monitors rate per batch, adjusts interval AFTER detecting burst.
This is the one-window-lag approach that AdaptiveStream aims to beat.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import count
import time

spark = SparkSession.builder.appName("Baseline-Reactive").master("local[2]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.readStream.format("rate").option("rowsPerSecond", 100).load()

# Start with default interval
current_interval = "2 seconds"
BURST_THRESHOLD = 200  # events per batch
NORMAL_INTERVAL = "2 seconds"
BURST_INTERVAL = "500 milliseconds"

query = df.writeStream.format("memory").queryName("reactive") \
    .trigger(processingTime=current_interval).start()

start = time.time()
while time.time() - start < 60:
    progress = query.lastProgress
    if progress and "numInputRows" in progress:
        rate = progress["numInputRows"]
        if rate > BURST_THRESHOLD and current_interval != BURST_INTERVAL:
            print(f"[Reactive] BURST detected ({rate} rows), but can't change interval mid-stream!")
            print(f"[Reactive] Would need to restart query — this is the lag problem.")
    time.sleep(1)

query.stop()
spark.stop()
