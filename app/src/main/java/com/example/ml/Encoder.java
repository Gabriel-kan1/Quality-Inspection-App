package com.example.ml;

import android.graphics.Bitmap;

/**
 * Turns an image into an embedding vector.
 *
 * <p>Exists so {@link AlignmentGate} can be exercised without a live ONNX session. The gate's
 * most consequential behaviour — that the scale probe is skipped for frames already below the
 * match threshold, which is what keeps it to one encoder pass on the frames the operator
 * spends most of their time on — was unreachable from a JVM test while the gate held a
 * concrete {@link FeatureEmbedder}.
 *
 * <p>Deliberately one method. This is a test seam, not an abstraction layer: the app has
 * exactly one encoder, and {@link FeatureEmbedder#ENCODER_ID} is still the single source of
 * truth for which embedding space a stored vector belongs to.
 */
public interface Encoder {

    /**
     * Embeds one image, or returns null when the encode failed. Callers treat null as "no
     * encoder available" and degrade rather than crashing.
     */
    float[] embed(Bitmap crop);
}
