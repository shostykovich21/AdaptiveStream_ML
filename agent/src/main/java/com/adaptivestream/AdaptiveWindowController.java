package com.adaptivestream;

import java.io.*;
import java.net.Socket;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicBoolean;

public class AdaptiveWindowController {

    // Shared interval — Spark reads this on every trigger via IntervalAdvice
    private static final AtomicLong currentIntervalMs = new AtomicLong(-1); // -1 = use default
    private static final AtomicBoolean fallbackMode = new AtomicBoolean(false);

    // Config
    private static final long UPDATE_PERIOD_MS = 1000;       // query LSTM every 1s
    private static final long MIN_INTERVAL_MS = 100;          // floor
    private static final long MAX_INTERVAL_MS = 30000;        // ceiling
    private static final long FALLBACK_INTERVAL_MS = 2000;    // safe default on failure
    private static final int PREDICTOR_PORT = 9876;
    private static final int MAX_RECONNECT_ATTEMPTS = 5;

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
                System.out.println("[Controller] Starting LSTM predictor subprocess...");
                try {
                    startPredictor();
                    Thread.sleep(3000); // wait for predictor to warm up

                    Socket socket = connectToPredictor();
                    if (socket == null) {
                        enterFallback("Could not connect to predictor");
                        return;
                    }

                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(socket.getInputStream()));
                    PrintWriter writer = new PrintWriter(
                        socket.getOutputStream(), true);

                    System.out.println("[Controller] Connected to predictor. Control loop running.");
                    int consecutiveErrors = 0;

                    while (!Thread.interrupted()) {
                        try {
                            // Send rate query
                            writer.println("predict");
                            String response = reader.readLine();

                            if (response == null) {
                                throw new IOException("Predictor closed connection");
                            }

                            // Parse response: "predicted_rate:450.5,confidence:0.92"
                            double predictedRate = parsePredictedRate(response);
                            double confidence = parseConfidence(response);

                            // Compute interval from predicted rate
                            long interval = computeInterval(predictedRate, confidence);
                            currentIntervalMs.set(interval);
                            fallbackMode.set(false);
                            consecutiveErrors = 0;

                        } catch (Exception e) {
                            consecutiveErrors++;
                            System.err.println("[Controller] Error: " + e.getMessage()
                                + " (attempt " + consecutiveErrors + ")");

                            if (consecutiveErrors >= 3) {
                                enterFallback("Too many consecutive errors");
                                // Try to restart predictor
                                restartPredictor();
                                socket = connectToPredictor();
                                if (socket != null) {
                                    reader = new BufferedReader(
                                        new InputStreamReader(socket.getInputStream()));
                                    writer = new PrintWriter(
                                        socket.getOutputStream(), true);
                                    consecutiveErrors = 0;
                                    System.out.println("[Controller] Predictor restarted successfully.");
                                }
                            }
                        }

                        Thread.sleep(UPDATE_PERIOD_MS);
                    }
                } catch (Exception e) {
                    enterFallback("Controller thread crashed: " + e.getMessage());
                }
            }
        }, "AdaptiveStream-Controller");

        controllerThread.setDaemon(true);
        controllerThread.start();
    }

    private static long computeInterval(double predictedRate, double confidence) {
        // High rate → short interval (process faster)
        // Low rate → long interval (save overhead)
        // Low confidence → conservative (use fallback interval)
        if (confidence < 0.5) {
            return FALLBACK_INTERVAL_MS;
        }

        // Target: process ~1000 events per window
        long interval = (long)(1000.0 / Math.max(predictedRate, 1.0) * 1000);

        // Clamp to bounds
        interval = Math.max(MIN_INTERVAL_MS, Math.min(interval, MAX_INTERVAL_MS));
        return interval;
    }

    private static void enterFallback(String reason) {
        System.out.println("[Controller] FALLBACK MODE: " + reason);
        currentIntervalMs.set(FALLBACK_INTERVAL_MS);
        fallbackMode.set(true);
    }

    private static void startPredictor() throws IOException {
        ProcessBuilder pb = new ProcessBuilder(
            "python3", "/home/aayushvbarhate/adaptivestream/predictor/predictor_server.py");
        pb.redirectErrorStream(true);
        predictorProcess = pb.start();
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
            System.err.println("[Controller] Failed to restart predictor: " + e.getMessage());
        }
    }

    private static Socket connectToPredictor() {
        for (int i = 0; i < MAX_RECONNECT_ATTEMPTS; i++) {
            try {
                return new Socket("127.0.0.1", PREDICTOR_PORT);
            } catch (Exception e) {
                try { Thread.sleep(1000); } catch (InterruptedException ie) { break; }
            }
        }
        return null;
    }

    private static double parsePredictedRate(String response) {
        // Format: "predicted_rate:450.5,confidence:0.92"
        try {
            String[] parts = response.split(",");
            return Double.parseDouble(parts[0].split(":")[1]);
        } catch (Exception e) {
            return 100.0; // safe default
        }
    }

    private static double parseConfidence(String response) {
        try {
            String[] parts = response.split(",");
            return Double.parseDouble(parts[1].split(":")[1]);
        } catch (Exception e) {
            return 0.0; // low confidence triggers fallback
        }
    }
}
