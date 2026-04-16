"""Fixed window baseline — static interval, no adaptation."""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, window, current_timestamp
import time, sys

INTERVAL = sys.argv[1] if len(sys.argv) > 1 else "2 seconds"
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 60

spark = SparkSession.builder.appName("Baseline-Fixed").master("local[2]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.readStream.format("rate").option("rowsPerSecond", 100).load()
agg = df.groupBy(window("timestamp", INTERVAL)).agg(count("*").alias("events"))

query = agg.writeStream.format("console").outputMode("complete") \
    .trigger(processingTime=INTERVAL).start()

time.sleep(DURATION)
query.stop()
spark.stop()
