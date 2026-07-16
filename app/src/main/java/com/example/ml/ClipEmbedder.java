package com.example.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * On-device CLIP image encoder (ViT-B/32), mirroring get_embedding() in
 * python/EdgeComputingImprove.py. Wraps an ONNX Runtime session over a bundled
 * CLIP vision-encoder model and turns a crop Bitmap into an embedding vector.
 *
 * Model asset: assets/{@link #MODEL_ASSET}. Expected input: a single float32
 * tensor of shape [1, 3, 224, 224] (NCHW, CLIP-normalised). The output vector
 * can be the projected image_embeds (512-d) or a pooled hidden state (768-d) —
 * either works, because the reference library is built with THIS SAME encoder,
 * so only self-consistency matters (not parity with desktop sentence-transformers).
 *
 * To (re)generate the model: export the vision tower of openai/clip-vit-base-patch32
 * to ONNX (e.g. optimum-cli / transformers.js exports), optionally int8-quantise,
 * and place it at app/src/main/assets/{@link #MODEL_ASSET}.
 */
public final class ClipEmbedder {

    private static final String TAG = "ClipEmbedder";
    public static final String MODEL_ASSET = "models/clip_image_encoder.onnx";

    private static final int INPUT_SIZE = 224;
    // CLIP image normalisation constants (OpenAI CLIP preprocessing).
    private static final float[] MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    private static final float[] STD  = {0.26862954f, 0.26130258f, 0.27577711f};

    private static volatile ClipEmbedder instance;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final String outputName;

    private ClipEmbedder(OrtEnvironment env, OrtSession session, String inputName, String outputName) {
        this.env = env;
        this.session = session;
        this.inputName = inputName;
        this.outputName = outputName;
    }

    /**
     * Returns the shared embedder, or null if the model asset is missing or fails
     * to load. Callers should treat null as "no CLIP available" and degrade to
     * an "unknown" classification rather than crashing.
     */
    public static ClipEmbedder get(Context context) {
        ClipEmbedder local = instance;
        if (local != null) return local;
        synchronized (ClipEmbedder.class) {
            if (instance == null) {
                instance = create(context.getApplicationContext());
            }
            return instance;
        }
    }

    private static ClipEmbedder create(Context context) {
        try {
            byte[] modelBytes = readAsset(context, MODEL_ASSET);
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            OrtSession session = env.createSession(modelBytes, opts);
            String inputName = session.getInputNames().iterator().next();
            String outputName = pickOutputName(session.getOutputNames());
            Log.d(TAG, "CLIP model loaded. input=" + inputName
                    + " outputs=" + session.getOutputNames() + " using=" + outputName);
            return new ClipEmbedder(env, session, inputName, outputName);
        } catch (Throwable t) {
            Log.e(TAG, "Failed to load CLIP model asset '" + MODEL_ASSET
                    + "'. Classification will fall back to 'unknown'.", t);
            return null;
        }
    }

    /**
     * Encodes a crop into an embedding vector. Thread-safe (synchronized) so it can
     * be called from the classification loop on a background thread. Returns null on
     * failure.
     */
    public synchronized float[] embed(Bitmap crop) {
        if (crop == null) return null;
        OnnxTensor input = null;
        OrtSession.Result result = null;
        try {
            FloatBuffer buffer = preprocess(crop);
            input = OnnxTensor.createTensor(
                    env, buffer, new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, input);
            result = session.run(inputs);
            return extractVector(result);
        } catch (Throwable t) {
            Log.e(TAG, "embed() failed", t);
            return null;
        } finally {
            if (result != null) result.close();
            if (input != null) input.close();
        }
    }

    /** Resize (shorter side -> 224), centre-crop 224x224, normalise, pack as NCHW. */
    private FloatBuffer preprocess(Bitmap src) {
        int w = src.getWidth();
        int h = src.getHeight();
        float scale = (float) INPUT_SIZE / Math.min(w, h);
        int rw = Math.max(INPUT_SIZE, Math.round(w * scale));
        int rh = Math.max(INPUT_SIZE, Math.round(h * scale));

        Bitmap resized = Bitmap.createScaledBitmap(src, rw, rh, true);
        int offX = (rw - INPUT_SIZE) / 2;
        int offY = (rh - INPUT_SIZE) / 2;

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, offX, offY, INPUT_SIZE, INPUT_SIZE);
        if (resized != src) resized.recycle();

        int area = INPUT_SIZE * INPUT_SIZE;
        float[] chw = new float[3 * area];
        for (int i = 0; i < area; i++) {
            int p = pixels[i];
            float r = ((p >> 16) & 0xFF) / 255f;
            float g = ((p >> 8) & 0xFF) / 255f;
            float b = (p & 0xFF) / 255f;
            chw[i]            = (r - MEAN[0]) / STD[0];          // R plane
            chw[area + i]     = (g - MEAN[1]) / STD[1];          // G plane
            chw[2 * area + i] = (b - MEAN[2]) / STD[2];          // B plane
        }
        // ONNX Runtime expects a direct buffer for zero-copy tensor creation.
        FloatBuffer fb = ByteBuffer
                .allocateDirect(chw.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        fb.put(chw);
        fb.rewind();
        return fb;
    }

    /**
     * Picks the most embedding-like output so different CLIP vision exports all work:
     * prefer a projected "image_embeds", then a pooled output, else the first output.
     */
    private static String pickOutputName(java.util.Set<String> outputs) {
        for (String name : outputs) {
            if (name.toLowerCase().contains("image_embeds")) return name;
        }
        for (String name : outputs) {
            String l = name.toLowerCase();
            if (l.contains("pooler") || l.contains("pooled") || l.contains("embed")) return name;
        }
        return outputs.iterator().next();
    }

    /** Reads the chosen output tensor and flattens it to a float[D] vector. */
    private float[] extractVector(OrtSession.Result result) throws Exception {
        OnnxValue value = result.get(outputName).orElseGet(() -> result.get(0));
        Object raw = value.getValue();
        if (raw instanceof float[][]) {
            // [1, D] — projected/pooled embedding.
            return ((float[][]) raw)[0];
        }
        if (raw instanceof float[]) {
            return (float[]) raw;
        }
        if (raw instanceof float[][][]) {
            // [1, tokens, D] last_hidden_state — use the CLS token (index 0).
            return ((float[][][]) raw)[0][0];
        }
        throw new IllegalStateException("Unsupported CLIP output type: " + raw.getClass());
    }

    private static byte[] readAsset(Context context, String path) throws Exception {
        try (InputStream is = context.getAssets().open(path)) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream(Math.max(1024, is.available()));
            byte[] buf = new byte[16 * 1024];
            int n;
            while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
            return bos.toByteArray();
        }
    }
}
