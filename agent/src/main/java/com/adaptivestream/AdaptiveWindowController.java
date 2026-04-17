package com.adaptivestream;

import java.io.*;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicBoolean;

public class AdaptiveWindowController {

    private static final AtomicLong currentIntervalMs = new AtomicLong(-1);
    private static final AtomicBoolean fallbackMode = new AtomicBoolean(false);
    private static final IntervalSmoother smoother = new IntervalSmoother();

    // Tunable parameters
    private static final long UPDATE_PERIOD_MS = 1000;
    private static final long MIN_INTERVAL_MS = 100;
    private static final long MAX_INTERVAL_MS = 30000;
    private static final long FALLBACK_INTERVAL_MS = 2000;
    private static final int PREDICTOR_PORT = 9876;
    private static final int MAX_RECONNECT_ATTEMPTS = 5;
    private static final long TARGET_EVENTS_PER_WINDOW = 1000;

    // Configurable via system properties — set with -Dadaptivestream.predictor.model=tcn etc.
    private static final String PREDICTOR_MODEL =
        System.getProperty("adaptivestream.predictor.model", "lstm");
    private static final String PYTHON_EXEC =
        System.getProperty("adaptivestream.python", "python3");
    private static final String PREDICTOR_SCRIPT = detectPredictorScript();

    private static Process predictorProcess;
    private static Thread controllerThread;

    public static long getCurrentIntervalMs() {
        return currentIntervalMs.get();
    }

    public static boolean isInFallback() {
        return fallbackMode.get();
    }

    public static void start() {
        controllerThread = new Thread(new Runnable() {
            public void run() {
                System.out.println("[Controller] Starting predictor subprocess ("
                    + PREDICTOR_MODEL + ") ...");
                try {
                    startPredictor();
                    Thread.sleep(3000);

                    Socket socket = connectToPredictor();
                    if (socket == null) {
                        enterFallback("Could not connect to predictor");
                        return;
                    }

                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(socket.getInputStream()));
                    PrintWriter writer = new PrintWriter(
                        socket.getOutputStream(), true);

                    System.out.println("[Controller] Connected. Control loop active.");
                    int consecutiveErrors = 0;

                    while (!Thread.interrupted()) {
                        try {
                            writer.println("predict");
                            String response = reader.readLine();

                            if (response == null) {
                                throw new IOException("Predictor closed connection");
                            }

                            double predictedRate = parsePredictedRate(response);
                            double confidence = parseConfidence(response);

                            long rawInterval = computeInterval(predictedRate, confidence);
                            long smoothedInterval = smoother.smooth(rawInterval);

                            currentIntervalMs.set(smoothedInterval);
                            fallbackMode.set(false);
                            consecutiveErrors = 0;

                            if (Math.abs(rawInterval - smoothedInterval) > 100) {
                                System.out.println("[Controller] Smoothed interval: "
                                    + rawInterval + "ms -> " + smoothedInterval
                                    + "ms (rate=" + String.format("%.0f", predictedRate)
                                    + ", conf=" + String.format("%.2f", confidence) + ")");
                            }

                        } catch (Exception e) {
                            consecutiveErrors++;
                            System.err.println("[Controller] Error: " + e.getMessage()
                                + " (" + consecutiveErrors + "/3)");

                            if (consecutiveErrors >= 3) {
                                enterFallback("Too many errors, restarting predictor");
                                restartPredictor();
                                socket = connectToPredictor();
                                if (socket != null) {
                                    reader = new BufferedReader(
                                        new InputStreamReader(socket.getInputStream()));
                                    writer = new PrintWriter(
                                        socket.getOutputStream(), true);
                                    consecutiveErrors = 0;
                                    System.out.println("[Controller] Predictor recovered.");
                                }
                            }
                        }

                        Thread.sleep(UPDATE_PERIOD_MS);
                    }
                } catch (Exception e) {
                    enterFallback("Controller crashed: " + e.getMessage());
                }
            }
        }, "AdaptiveStream-Controller");

        controllerThread.setDaemon(true);
        controllerThread.start();
    }

    /**
     * Compute target interval from predicted rate using adaptive formula.
     *
     * Core idea: target a fixed number of events per window (TARGET_EVENTS_PER_WINDOW).
     *   interval = TARGET_EVENTS / predicted_rate
     *
     * Confidence scaling: low confidence → blend toward fallback interval.
     *   effective_interval = confidence * computed + (1-confidence) * fallback
     *
     * This replaces the naive hardcoded heuristic with a principled approach:
     * - High rate (burst): short interval → process faster, reduce latency
     * - Low rate (quiet): long interval → fewer micro-batches, save overhead
     * - Low confidence: conservative → use safe fallback interval
     */
    static long computeInterval(double predictedRate, double confidence) {
        if (confidence < 0.1 || predictedRate <= 0) {
            return FALLBACK_INTERVAL_MS;
        }

        // Target-based computation
        double computedSec = TARGET_EVENTS_PER_WINDOW / Math.max(predictedRate, 1.0);
        long computedMs = (long)(computedSec * 1000);

        // Confidence-weighted blending with fallback
        long blended = (long)(confidence * computedMs + (1.0 - confidence) * FALLBACK_INTERVAL_MS);

        // Clamp to operational bounds
        return Math.max(MIN_INTERVAL_MS, Math.min(blended, MAX_INTERVAL_MS));
    }

    private static void enterFallback(String reason) {
        System.out.println("[Controller] FALLBACK: " + reason);
        smoother.reset(FALLBACK_INTERVAL_MS);
        currentIntervalMs.set(FALLBACK_INTERVAL_MS);
        fallbackMode.set(true);
    }

    /**
     * Locate predictor_server.py portably.
     * Priority: system property > relative to agent JAR > relative to working dir > env var.
     */
    private static String detectPredictorScript() {
        String prop = System.getProperty("adaptivestream.predictor.script");
        if (prop != null) return prop;

        // JAR lives in agent/target/ — go up two levels to project root, then into predictor/
        try {
            String jarPath = AdaptiveWindowController.class
                .getProtectionDomain().getCodeSource().getLocation().toURI().getPath();
            Path script = Paths.get(jarPath).getParent().getParent().getParent()
                .resolve("predictor/predictor_server.py").normalize();
            if (Files.exists(script)) return script.toAbsolutePath().toString();
        } catch (Exception ignored) {}

        // Working directory fallback
        Path wdScript = Paths.get("predictor/predictor_server.py");
        if (Files.exists(wdScript)) return wdScript.toAbsolutePath().toString();

        // Environment variable fallback
        String env = System.getenv("ADAPTIVESTREAM_PREDICTOR");
        if (env != null) return env;

        return "predictor/predictor_server.py";
    }

    private static void startPredictor() throws IOException {
        System.out.println("[Controller] Script: " + PREDICTOR_SCRIPT);
        System.out.println("[Controller] Model:  " + PREDICTOR_MODEL);
        ProcessBuilder pb = new ProcessBuilder(
            PYTHON_EXEC, PREDICTOR_SCRIPT, "--model", PREDICTOR_MODEL);
        pb.redirectErrorStream(true);
        predictorProcess = pb.start();

        // Log predictor output in background
        final InputStream is = predictorProcess.getInputStream();
        Thread logThread = new Thread(new Runnable() {
            public void run() {
                try {
                    BufferedReader r = new BufferedReader(new InputStreamReader(is));
                    String line;
                    while ((line = r.readLine()) != null) {
                        System.out.println(line);
                    }
                } catch (Exception e) {}
            }
        }, "Predictor-Logger");
        logThread.setDaemon(true);
        logThread.start();
    }

    private static void restartPredictor() {
        try {
            if (predictorProcess != null) {
                predictorProcess.destroyForcibly();
                predictorProcess.waitFor();
            }
            Thread.sleep(1000);
            startPredictor();
            Thread.sleep(3000);
        } catch (Exception e) {
            System.err.println("[Controller] Restart failed: " + e.getMessage());
        }
    }

    private static Socket connectToPredictor() {
        for (int i = 0; i < MAX_RECONNECT_ATTEMPTS; i++) {
            try {
                return new Socket("127.0.0.1", PREDICTOR_PORT);
            } catch (Exception e) {
                System.out.println("[Controller] Waiting for predictor... (" + (i+1) + ")");
                try { Thread.sleep(1000); } catch (InterruptedException ie) { break; }
            }
        }
        return null;
    }

    private static double parsePredictedRate(String response) {
        try {
            for (String part : response.split(",")) {
                if (part.startsWith("predicted_rate:")) {
                    return Double.parseDouble(part.split(":")[1]);
                }
            }
        } catch (Exception e) {}
        return 100.0;
    }

    private static double parseConfidence(String response) {
        try {
            for (String part : response.split(",")) {
                if (part.startsWith("confidence:")) {
                    return Double.parseDouble(part.split(":")[1]);
                }
            }
        } catch (Exception e) {}
        return 0.0;
    }
}
