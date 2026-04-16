package com.adaptivestream;

import net.bytebuddy.asm.Advice;

public class IntervalAdvice {
    @Advice.OnMethodExit
    public static void onExit(@Advice.Return(readOnly = false) long returnValue) {
        long override = AdaptiveWindowController.getCurrentIntervalMs();
        if (override > 0) {
            returnValue = override;
        }
        // If override <= 0, keep original value (fallback to static)
    }
}
