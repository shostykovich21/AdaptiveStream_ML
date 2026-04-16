package com.adaptivestream;

import org.junit.Test;
import static org.junit.Assert.*;

public class IntervalSmootherTest {

    @Test
    public void testFirstCallReturnsTarget() {
        IntervalSmoother s = new IntervalSmoother();
        assertEquals(2000, s.smooth(2000));
    }

    @Test
    public void testSmoothingPreventsJumps() {
        IntervalSmoother s = new IntervalSmoother();
        s.smooth(2000);  // initialize

        // Sudden drop from 2000 to 100 — should be rate-limited
        long result = s.smooth(100);
        assertTrue("Should not drop below half: got " + result, result >= 1000);
    }

    @Test
    public void testSmoothingPreventsSpikes() {
        IntervalSmoother s = new IntervalSmoother();
        s.smooth(500);  // initialize

        // Sudden jump from 500 to 20000 — should be rate-limited
        long result = s.smooth(20000);
        assertTrue("Should not exceed double: got " + result, result <= 1000);
    }

    @Test
    public void testGradualConvergence() {
        IntervalSmoother s = new IntervalSmoother();
        s.smooth(2000);

        // Repeatedly request 500ms — should gradually converge
        long prev = 2000;
        for (int i = 0; i < 20; i++) {
            long val = s.smooth(500);
            assertTrue("Should be decreasing or stable", val <= prev);
            prev = val;
        }
        // After 20 iterations should be close to 500
        assertTrue("Should converge near 500: got " + prev, prev < 700);
    }

    @Test
    public void testFloorEnforced() {
        IntervalSmoother s = new IntervalSmoother();
        s.smooth(200);

        // Even with very low target, never go below 100ms
        for (int i = 0; i < 50; i++) {
            long val = s.smooth(1);
            assertTrue("Never below 100ms: got " + val, val >= 100);
        }
    }

    @Test
    public void testCeilingEnforced() {
        IntervalSmoother s = new IntervalSmoother();
        s.smooth(15000);

        for (int i = 0; i < 50; i++) {
            long val = s.smooth(999999);
            assertTrue("Never above 30s: got " + val, val <= 30000);
        }
    }

    @Test
    public void testResetOverrides() {
        IntervalSmoother s = new IntervalSmoother();
        s.smooth(2000);
        s.smooth(2000);

        s.reset(5000);
        assertEquals(5000, s.getCurrentSmoothed());
    }
}
