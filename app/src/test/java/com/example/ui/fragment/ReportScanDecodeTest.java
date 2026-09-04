package com.example.ui.fragment;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import android.graphics.Bitmap;

import com.example.data.entity.InspectionRecord;
import com.example.report.ReportCode;
import com.google.zxing.BinaryBitmap;
import com.google.zxing.DecodeHintType;
import com.google.zxing.MultiFormatReader;
import com.google.zxing.PlanarYUVLuminanceSource;
import com.google.zxing.Result;
import com.google.zxing.common.HybridBinarizer;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.GraphicsMode;

import java.nio.ByteBuffer;
import java.util.EnumMap;
import java.util.Map;
import java.util.UUID;

/**
 * Covers the frame-handling half of the scanner — the part a camera would exercise and a unit
 * test otherwise would not.
 *
 * <p>Camera hardware does not hand over a tidy width×height buffer. Each row is padded out to
 * the hardware's own stride, and some devices interleave the luminance samples so that reading
 * consecutive bytes walks across colour channels instead of along a row. Getting that arithmetic
 * wrong throws nothing: it yields a sheared image that never decodes, which on the screen is
 * indistinguishable from "the camera cannot see the code". So the padded and interleaved layouts
 * are reproduced here deliberately.
 */
@RunWith(RobolectricTestRunner.class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
public class ReportScanDecodeTest {

    // ── Stride unpacking ────────────────────────────────────────────────────

    @Test
    public void packLuminanceStripsRowPadding() {
        int width = 4, height = 3, rowStride = 7;
        byte[] padded = new byte[rowStride * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                padded[y * rowStride + x] = (byte) (y * 10 + x);
            }
            // Junk in the padding, which is what real hardware leaves there.
            for (int x = width; x < rowStride; x++) {
                padded[y * rowStride + x] = (byte) 0xEE;
            }
        }

        byte[] out = ReportScanFragment.packLuminance(
                ByteBuffer.wrap(padded), width, height, rowStride, 1);

        assertArrayEquals(new byte[]{0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23}, out);
    }

    @Test
    public void packLuminanceHandlesInterleavedSamples() {
        int width = 3, height = 2, pixelStride = 2, rowStride = width * pixelStride + 2;
        byte[] interleaved = new byte[rowStride * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                interleaved[y * rowStride + x * pixelStride] = (byte) (y * 10 + x);
                interleaved[y * rowStride + x * pixelStride + 1] = (byte) 0xEE;
            }
        }

        byte[] out = ReportScanFragment.packLuminance(
                ByteBuffer.wrap(interleaved), width, height, rowStride, pixelStride);

        assertArrayEquals(new byte[]{0, 1, 2, 10, 11, 12}, out);
    }

    @Test
    public void packLuminanceIsIdentityWhenThereIsNoPadding() {
        int width = 4, height = 2;
        byte[] tight = {0, 1, 2, 3, 4, 5, 6, 7};

        byte[] out = ReportScanFragment.packLuminance(
                ByteBuffer.wrap(tight), width, height, width, 1);

        assertArrayEquals(tight, out);
    }

    // ── The full frame path ─────────────────────────────────────────────────

    /**
     * A rendered code, laid out the way a camera would deliver it, read back through exactly
     * the source and binarizer the analyser uses.
     */
    @Test
    public void aCodeInACameraShapedFrameDecodes() throws Exception {
        InspectionRecord record = new InspectionRecord();
        record.setUuid("3f2504e0-4f89-41d3-9a0c-0305e82c3301");
        record.setPassOrFail("FAIL");
        String payload = ReportCode.encode(record);

        byte[] frame = cameraFrame(ReportCode.toQrBitmap(payload, 360), 640, 480, 704, 1);

        assertEquals(payload, decodeFrame(frame, 640, 480));
    }

    /** The same frame with interleaved samples, which is what several real devices produce. */
    @Test
    public void aCodeDecodesFromAnInterleavedFrame() throws Exception {
        InspectionRecord record = new InspectionRecord();
        record.setUuid(UUID.randomUUID().toString());
        record.setPassOrFail("PASS");
        String payload = ReportCode.encode(record);

        byte[] frame = cameraFrame(ReportCode.toQrBitmap(payload, 360), 640, 480, 1408, 2);
        String scanned = decodeFrame(frame, 640, 480);

        assertEquals(payload, scanned);
        ReportCode.Pointer pointer = ReportCode.decode(scanned);
        assertNotNull(pointer);
        assertEquals(record.getUuid(), pointer.uuid);
        assertEquals("PASS", pointer.verdictText());
    }

    /**
     * Composites the QR onto a grey background at camera resolution, then lays that out with the
     * given stride and pixel stride and unpacks it through the production code.
     */
    private static byte[] cameraFrame(Bitmap qr, int width, int height,
                                      int rowStride, int pixelStride) {
        byte[] scene = new byte[width * height];
        java.util.Arrays.fill(scene, (byte) 0xB0);

        int size = qr.getWidth();
        int left = (width - size) / 2;
        int top = (height - size) / 2;
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                // The bitmap is black-on-white, so the blue channel doubles as luminance.
                int luma = qr.getPixel(x, y) & 0xFF;
                scene[(top + y) * width + (left + x)] = (byte) luma;
            }
        }

        byte[] strided = new byte[rowStride * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                strided[y * rowStride + x * pixelStride] = scene[y * width + x];
            }
        }
        return ReportScanFragment.packLuminance(
                ByteBuffer.wrap(strided), width, height, rowStride, pixelStride);
    }

    private static String decodeFrame(byte[] luminance, int width, int height) throws Exception {
        Map<DecodeHintType, Object> hints = new EnumMap<>(DecodeHintType.class);
        hints.put(DecodeHintType.TRY_HARDER, Boolean.TRUE);
        MultiFormatReader reader = new MultiFormatReader();
        reader.setHints(hints);

        PlanarYUVLuminanceSource source = new PlanarYUVLuminanceSource(
                luminance, width, height, 0, 0, width, height, false);
        Result result = reader.decodeWithState(new BinaryBitmap(new HybridBinarizer(source)));
        return result.getText();
    }
}
