package com.adaptivestream;

import net.bytebuddy.agent.builder.AgentBuilder;
import net.bytebuddy.asm.Advice;
import net.bytebuddy.matcher.ElementMatchers;
import java.lang.instrument.Instrumentation;

public class AdaptiveStreamAgent {
    public static void premain(String args, Instrumentation inst) {
        System.out.println("[AdaptiveStream] Agent v1.0 loading...");

        // Start controller (manages predictor subprocess + interval updates)
        AdaptiveWindowController.start();

        // Instrument Spark's trigger interval
        new AgentBuilder.Default()
            .type(ElementMatchers.named(
                "org.apache.spark.sql.execution.streaming.ProcessingTimeExecutor"))
            .transform(new AgentBuilder.Transformer.ForAdvice()
                .advice(ElementMatchers.named("intervalMs"),
                        "com.adaptivestream.IntervalAdvice"))
            .with(new AgentBuilder.Listener.StreamWriting(System.err)
                .withErrorsOnly())
            .installOn(inst);

        System.out.println("[AdaptiveStream] Instrumentation active.");
    }
}
