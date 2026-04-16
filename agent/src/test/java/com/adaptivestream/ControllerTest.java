package com.adaptivestream;

import org.junit.Test;
import static org.junit.Assert.*;

public class ControllerTest {

    @Test
    public void testHighRateGivesShortInterval() {
        // 1000 events/sec → target 1000 events per window → 1 second
        long interval = AdaptiveWindowController.computeInterval(1000.0, 0.9);
        assertTrue("High rate should give short interval: got " + interval, interval < 2000);
        assertTrue("Should be at least 100ms: got " + interval, interval >= 100);
    }

    @Test
    public void testLowRateGivesLongInterval() {
        // 10 events/sec → target 1000 events per window → 100 seconds (clamped to 30s)
        long interval = AdaptiveWindowController.computeInterval(10.0, 0.9);
        assertTrue("Low rate should give long interval: got " + interval, interval > 5000);
    }

    @Test
    public void testVeryHighRateClampedToMin() {
        // 100000 events/sec → 10ms computed → clamped to 100ms
        long interval = AdaptiveWindowController.computeInterval(100000.0, 1.0);
        assertEquals("Should clamp to minimum", 100, interval);
    }

    @Test
    public void testZeroConfidenceReturnsFallback() {
        long interval = AdaptiveWindowController.computeInterval(500.0, 0.0);
        assertEquals("Zero confidence should use fallback", 2000, interval);
    }

    @Test
    public void testLowConfidenceBlendsTowardFallback() {
        long highConf = AdaptiveWindowController.computeInterval(5000.0, 0.9);
        long lowConf = AdaptiveWindowController.computeInterval(5000.0, 0.3);
        assertTrue("Low confidence should be closer to fallback (2000ms): low=" + lowConf + " high=" + highConf,
            Math.abs(lowConf - 2000) < Math.abs(highConf - 2000));
    }

    @Test
    public void testNegativeRateReturnsFallback() {
        long interval = AdaptiveWindowController.computeInterval(-50.0, 0.8);
        assertEquals("Negative rate should use fallback", 2000, interval);
    }

    @Test
    public void testZeroRateReturnsFallback() {
        long interval = AdaptiveWindowController.computeInterval(0.0, 0.8);
        assertEquals("Zero rate should use fallback", 2000, interval);
    }
}
