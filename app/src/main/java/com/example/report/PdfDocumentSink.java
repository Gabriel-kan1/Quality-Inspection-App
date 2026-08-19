package com.example.report;

import android.graphics.Canvas;
import android.graphics.pdf.PdfDocument;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.io.IOException;
import java.io.OutputStream;

/**
 * The real {@link PageSink}: pages go into an Android framework {@link PdfDocument}.
 *
 * <p>Deliberately does nothing but adapt. All the layout decisions live in
 * {@link PdfReportWriter}, because none of this class can run outside a device or emulator.
 */
public final class PdfDocumentSink implements PageSink {

    private final PdfDocument document = new PdfDocument();
    @Nullable private PdfDocument.Page page;

    @NonNull
    @Override
    public Canvas beginPage(int pageNumber, int widthPt, int heightPt) {
        PdfDocument.PageInfo info =
                new PdfDocument.PageInfo.Builder(widthPt, heightPt, pageNumber).create();
        page = document.startPage(info);
        return page.getCanvas();
    }

    @Override
    public void finishPage() {
        if (page == null) return;
        document.finishPage(page);
        page = null;
    }

    @Override
    public void writeTo(@NonNull OutputStream out) throws IOException {
        try {
            document.writeTo(out);
        } finally {
            document.close();
        }
    }

    @Override
    public void discard() {
        if (page != null) {
            document.finishPage(page);
            page = null;
        }
        document.close();
    }
}
