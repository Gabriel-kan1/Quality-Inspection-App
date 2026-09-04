package com.example.report;

import android.graphics.Canvas;

import androidx.annotation.NonNull;

import java.io.IOException;
import java.io.OutputStream;

/**
 * Where {@link PdfReportWriter} puts the pages it lays out.
 *
 * <p>This seam exists for one reason: {@link android.graphics.pdf.PdfDocument} is a thin wrapper
 * over Skia's PDF backend that Robolectric implements no part of, so any code touching it
 * directly cannot be tested off-device. Everything worth testing here is the layout arithmetic —
 * where a row breaks, how tall a wrapped cell is, when a page fills up — and that only needs a
 * {@link Canvas}. Splitting the two lets the arithmetic be exercised on the JVM and leaves a
 * thin, obvious adapter ({@link PdfDocumentSink}) to confirm on a device.
 */
public interface PageSink {

    /** Starts a page and returns the canvas to draw it on. Coordinates are in points. */
    @NonNull
    Canvas beginPage(int pageNumber, int widthPt, int heightPt);

    /** Completes the page most recently begun. */
    void finishPage();

    /** Serialises the finished document. */
    void writeTo(@NonNull OutputStream out) throws IOException;

    /** Releases resources without producing output, for the failure path. */
    void discard();
}
