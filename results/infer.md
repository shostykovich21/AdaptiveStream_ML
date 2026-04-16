# What we learned from feasibility testing

These are our takeaways from running the initial round of tests on a GCP VM (2 cores, 4GB RAM, Ubuntu 24.04). All tests were done before writing any of the actual evaluation pipeline — the goal was to figure out if the core ideas even work before committing to a full implementation.

## 1. The agent approach works and it's not a hack

Our biggest worry was that Spark might cache `intervalMs()` internally and only read it once at startup. If that were the case, the whole bytecode instrumentation idea would be dead on arrival.

Turns out Spark calls `intervalMs()` on **every single trigger cycle**. We confirmed this by logging every interception — the method gets called multiple times per batch (see test1 output). This means we can change the return value at any point and Spark will pick it up on the next trigger without any restart or state reset.

ByteBuddy loaded cleanly via `-javaagent`, intercepted the right class (`ProcessingTimeExecutor` in `spark-sql_2.12-3.5.4.jar`), and the whole thing added basically zero overhead to the method call itself. This is the same pattern that OpenTelemetry and Datadog use in production, so we're not inventing anything sketchy here.

## 2. LSTM is worth the complexity

We initially thought a simple exponential moving average might be "good enough" and that LSTM would be hard to justify. After running the comparison (test6), that concern went away completely.

LSTM beats EMA by **89% on overall MAE** and by **94% during transitions** (the ramp-up/ramp-down moments that actually matter for window sizing). The reason is straightforward: EMA and SMA can only react to recent values. When a burst starts ramping up, they're always a few steps behind. LSTM learns the shape of bursts from training data — it recognizes the ramp-up pattern and predicts ahead.

The one caveat is that this only holds with proper training. Our first attempt (50 series, 10 epochs, no normalization) actually had LSTM performing worse than EMA. Per-series normalization and enough training data (100+ series, 30 epochs) were essential. This is worth noting in the paper — LSTM isn't a magic drop-in, you need to train it right.

## 3. Inference is fast enough to not matter

We were worried about LSTM adding latency to the control loop. At 0.67ms mean inference time on CPU, this is a non-issue. Even the p99 is only 1ms. The controller queries the predictor every 1 second, so spending 1ms on inference is 0.1% of the cycle.

The one gotcha is PyTorch's cold start: the first inference takes ~178ms because of JIT compilation. We handle this by running 10 dummy inferences when the predictor starts up. After warmup, it's consistently sub-millisecond.

## 4. IPC adds negligible overhead

TCP socket roundtrip on localhost is ~1.2ms (including inference). We considered Unix domain sockets but TCP is simpler and the performance difference at localhost is irrelevant at this scale. The socket overhead is <0.3ms on top of inference time.

## 5. The agent adds measurable but acceptable overhead

The overhead test (test7) showed +7.6% median batch latency and +47% memory (82MB). The memory increase is ByteBuddy + the agent classes + the controller thread. The latency overhead in steady state is small — the p95 spike (+113%) is from the first few batches where ByteBuddy is doing classload-time instrumentation. After that it settles down.

For a production system, 82MB extra memory is nothing. 7.6% batch latency is acceptable, especially since the whole point is to reduce latency during bursts by a much larger margin.

## 6. Window resizing needs to be smooth

This one we didn't test directly but realized while writing the controller: if the interval suddenly jumps from 10s to 500ms, Spark's internal state (watermarks, buffered events) could get confused. Events that were supposed to be in a 10s window might get split across multiple 500ms windows, or worse, get dropped as "late" by the watermark.

We built an `IntervalSmoother` that rate-limits transitions — the interval can at most halve or double per update cycle, and uses EMA smoothing on top of that. The unit tests verify convergence, floor/ceiling enforcement, and that it actually prevents sudden jumps. We haven't tested the actual Spark state behavior yet (that's a Semester 1 task), but the smoother should prevent the worst cases.

## 7. End-to-end actually works

The final integration test was the most satisfying. We started Spark with the agent, the agent automatically spawned the predictor subprocess, the controller connected via TCP, and Spark streamed data with the instrumented trigger interval. No manual coordination, no separate processes to manage.

The fallback logic also works — if the predictor dies, the controller reverts to a safe 2-second interval and tries to restart the subprocess. We haven't stress-tested this yet but the basic path is there.

## What's still unproven

- We haven't measured real end-to-end latency (producer timestamp → Spark output). The latency benchmark code is written but hasn't been run against actual burst traffic yet.
- LSTM was only trained and tested on synthetic data. NYC Taxi and Wikipedia EventStream evaluation is needed to prove generalization.
- We've only tested on Spark 3.5.4. The agent targets a specific internal class name that could change between versions.
- The confidence estimation (MC dropout) hasn't been validated — we're not sure if it actually gives low confidence on out-of-distribution patterns.

## Summary

The core technical bet — that we can intercept and dynamically override Spark's trigger interval from a JVM agent — is validated. The ML side (LSTM beats simpler baselines significantly) is also validated. What remains is evaluation on real data, scale testing, and the novelty contributions (multi-objective optimization, RL comparison, formal latency bounds). The foundation is solid.
