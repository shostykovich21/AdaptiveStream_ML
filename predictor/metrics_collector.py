"""
Collects real streaming rates from Spark's StreamingQueryProgress.
Spark exposes this via the StreamingQueryListener API or by polling query.lastProgress.

Two collection modes:
  1. Driver-side: polls query.lastProgress from the same JVM (used when predictor
     runs as subprocess of the Spark driver)
  2. REST API: scrapes Spark's REST API at :4040/api/v1/applications/.../streaming/statistics
     (used when predictor runs externally)
"""

import requests
import time
import threading
from collections import deque


class SparkMetricsCollector:
    """Collects inputRowsPerSecond from Spark's REST API."""

    def __init__(self, spark_ui_url="http://localhost:4040", window_size=30, poll_interval=1.0):
        self.spark_ui_url = spark_ui_url.rstrip("/")
        self.window_size = window_size
        self.poll_interval = poll_interval
        self.rates = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_loop(self):
        app_id = None
        while self._running:
            try:
                # Get active application ID
                if app_id is None:
                    apps = requests.get(f"{self.spark_ui_url}/api/v1/applications", timeout=2).json()
                    if apps:
                        app_id = apps[0]["id"]

                if app_id:
                    # Get streaming query progress
                    url = f"{self.spark_ui_url}/api/v1/applications/{app_id}/streaming/statistics"
                    resp = requests.get(url, timeout=2)

                    if resp.status_code == 200:
                        data = resp.json()
                        # Extract inputRowsPerSecond from recent batches
                        if "recentProgress" in data:
                            for progress in data["recentProgress"]:
                                rate = progress.get("inputRowsPerSecond", 0)
                                if rate > 0:
                                    self._add_rate(rate)
                    else:
                        # Spark 3.x: try individual query endpoints
                        url = f"{self.spark_ui_url}/api/v1/applications/{app_id}/sql/streaming"
                        resp = requests.get(url, timeout=2)
                        if resp.status_code == 200:
                            queries = resp.json()
                            for q in queries:
                                if "runId" in q:
                                    rate = q.get("inputRowsPerSecond", 0)
                                    if rate > 0:
                                        self._add_rate(rate)

            except requests.exceptions.ConnectionError:
                pass  # Spark UI not up yet, keep trying
            except Exception as e:
                print(f"[MetricsCollector] Error polling Spark: {e}")

            time.sleep(self.poll_interval)

    def _add_rate(self, rate):
        with self._lock:
            self.rates.append(float(rate))

    def add_rate_manual(self, rate):
        """For testing or manual rate injection."""
        self._add_rate(rate)

    def get_history(self):
        with self._lock:
            return list(self.rates)

    def is_ready(self):
        with self._lock:
            return len(self.rates) >= self.window_size


class DriverSideCollector:
    """
    Collects rates when running inside the Spark driver JVM.
    Controller pushes rates directly from StreamingQueryProgress callbacks.
    """

    def __init__(self, window_size=30):
        self.rates = deque(maxlen=window_size)
        self.window_size = window_size
        self._lock = threading.Lock()

    def add_rate(self, rate):
        with self._lock:
            self.rates.append(float(rate))

    def get_history(self):
        with self._lock:
            return list(self.rates)

    def is_ready(self):
        with self._lock:
            return len(self.rates) >= self.window_size
