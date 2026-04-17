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

    def __init__(self, spark_ui_url="http://localhost:4040", window_size=30,
                 poll_interval=1.0, app_name=None):
        self.spark_ui_url = spark_ui_url.rstrip("/")
        self.window_size  = window_size
        self.poll_interval = poll_interval
        self.app_name     = app_name   # if set, only attach to this Spark app name
        self.rates        = deque(maxlen=window_size)
        self._lock        = threading.Lock()
        self._running     = False
        self._thread      = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_loop(self):
        app_id              = None
        max_batch_id        = -1          # dedupe for recentProgress path
        seen_batch_keys     = set()       # dedupe for sql/streaming fallback: (runId, latestBatchId)
        while self._running:
            try:
                # Get active application ID
                if app_id is None:
                    apps = requests.get(f"{self.spark_ui_url}/api/v1/applications", timeout=2).json()
                    if apps:
                        if self.app_name:
                            matched = [a for a in apps if a.get("name") == self.app_name]
                            app_id = matched[0]["id"] if matched else None
                        else:
                            app_id = apps[0]["id"]

                if app_id:
                    # Get streaming query progress
                    url = f"{self.spark_ui_url}/api/v1/applications/{app_id}/streaming/statistics"
                    resp = requests.get(url, timeout=2)

                    if resp.status_code == 200:
                        data = resp.json()
                        # recentProgress returns ALL recent batches on every poll.
                        # batch IDs are monotonically increasing, so skip anything
                        # already seen by comparing against max_batch_id.
                        if "recentProgress" in data:
                            new_max = max_batch_id
                            for progress in data["recentProgress"]:
                                bid  = progress.get("batchId")
                                rate = progress.get("inputRowsPerSecond", 0)
                                if bid is None or bid <= max_batch_id:
                                    continue
                                new_max = max(new_max, bid)
                                self._add_rate(rate)  # record 0.0 — idle is a real observation
                            max_batch_id = new_max
                    else:
                        # Spark 3.x: try individual query endpoints.
                        # runId is a query-lifetime identifier — not a batch counter.
                        # Use (runId, latestBatchId) as the composite key so we
                        # collect one rate per batch, not one rate per query lifetime.
                        url = f"{self.spark_ui_url}/api/v1/applications/{app_id}/sql/streaming"
                        resp = requests.get(url, timeout=2)
                        if resp.status_code == 200:
                            for q in resp.json():
                                run_id   = q.get("runId")
                                batch_id = q.get("latestBatchId")
                                key      = (run_id, batch_id)
                                if run_id is None or batch_id is None or key in seen_batch_keys:
                                    continue
                                seen_batch_keys.add(key)
                                self._add_rate(q.get("inputRowsPerSecond", 0))  # record 0.0 too

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
