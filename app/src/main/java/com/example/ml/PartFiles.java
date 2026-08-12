package com.example.ml;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * Small shared helpers for the per-part files the learning loop writes.
 *
 * <p>Both of these existed in three or four copies across {@link ThresholdTuner},
 * {@link ReferenceLibraryProvider}, {@link LearningCropStore}, {@link ReferenceLibrary} and
 * {@link FeatureEmbedder}. The name sanitiser is the one that mattered: every copy has to
 * agree exactly, because they are all deriving a path for the <em>same</em> part, and a
 * divergence would silently point one subsystem at a different file than the others.
 */
public final class PartFiles {

    private PartFiles() {}

    /**
     * A part id reduced to something safe to use as a filename.
     *
     * <p>Part ids come from scanned barcodes and operator typing, so they can contain path
     * separators and anything else the scanner emits.
     */
    public static String safeName(String partId) {
        if (partId == null || partId.isEmpty()) return "unknown_part";
        return partId.replaceAll("[^A-Za-z0-9_.-]", "_");
    }

    /** Reads a stream fully into a byte array. Does not close it — callers use try-with-resources. */
    public static byte[] readAllBytes(InputStream is) throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int read;
        while ((read = is.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
        return out.toByteArray();
    }

    /** Reads a stream fully as UTF-8 text. */
    public static String readAllText(InputStream is) throws Exception {
        return new String(readAllBytes(is), StandardCharsets.UTF_8);
    }
}
