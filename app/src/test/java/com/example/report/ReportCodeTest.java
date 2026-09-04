package com.example.report;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import android.graphics.Bitmap;

import com.example.data.entity.InspectionRecord;
import com.google.zxing.BinaryBitmap;
import com.google.zxing.DecodeHintType;
import com.google.zxing.MultiFormatReader;
import com.google.zxing.RGBLuminanceSource;
import com.google.zxing.Result;
import com.google.zxing.common.HybridBinarizer;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.GraphicsMode;

import java.util.EnumMap;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

/**
 * Covers the pointer that gets printed on a report.
 *
 * <p>The round trip is the whole point: a code that encodes but does not decode back to the
 * same record is worse than no code, because it looks like traceability. So the tests go all
 * the way out to a rendered bitmap and back in through ZXing's own reader rather than stopping
 * at the byte packing.
 */
@RunWith(RobolectricTestRunner.class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
public class ReportCodeTest {

    private static final String UUID_TEXT = "3f2504e0-4f89-41d3-9a0c-0305e82c3301";
    private static final String DIGEST =
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

    private static InspectionRecord record(String verdict, String digest) {
        InspectionRecord r = new InspectionRecord();
        r.setUuid(UUID_TEXT);
        r.setPassOrFail(verdict);
        r.setDigest(digest);
        return r;
    }

    // ── Payload ─────────────────────────────────────────────────────────────

    @Test
    public void payloadIsFixedWidthWhateverTheRecord() {
        int sealed = ReportCode.encode(record("FAIL", DIGEST)).length();
        int unsealed = ReportCode.encode(record("PASS", null)).length();

        // The fixed width is the reason this carries a pointer and not a summary: a symbol
        // whose size depends on how many slots the part has is a symbol whose scan reliability
        // depends on the part.
        assertEquals(sealed, unsealed);
        assertEquals(ReportCode.PREFIX.length() + 42, sealed);
    }

    @Test
    public void roundTripsThroughThePayloadString() {
        ReportCode.Pointer pointer = ReportCode.decode(ReportCode.encode(record("FAIL", DIGEST)));

        assertNotNull(pointer);
        assertEquals(ReportCode.FORMAT_VERSION, pointer.formatVersion);
        assertEquals(UUID_TEXT, pointer.uuid);
        assertEquals("FAIL", pointer.verdictText());
        assertEquals(DIGEST.substring(0, ReportCode.DIGEST_PREFIX_BYTES * 2), pointer.digestPrefix);
    }

    @Test
    public void everyVerdictSurvivesTheRoundTrip() {
        for (String verdict : new String[]{"PASS", "FAIL", "UNVERIFIED"}) {
            ReportCode.Pointer pointer =
                    ReportCode.decode(ReportCode.encode(record(verdict, DIGEST)));
            assertNotNull(verdict, pointer);
            assertEquals(verdict, pointer.verdictText());
        }
    }

    /** An unsealed record has no digest, and the pointer has to say so rather than invent one. */
    @Test
    public void unsealedRecordDecodesWithNoDigest() {
        ReportCode.Pointer pointer = ReportCode.decode(ReportCode.encode(record("PASS", null)));

        assertNotNull(pointer);
        assertNull(pointer.digestPrefix);
        assertFalse(pointer.matches(record("PASS", null)));
    }

    @Test
    public void pointerMatchesItsOwnRecordAndNotAnAlteredOne() {
        InspectionRecord sealed = record("FAIL", DIGEST);
        ReportCode.Pointer pointer = ReportCode.decode(ReportCode.encode(sealed));

        assertNotNull(pointer);
        assertTrue(pointer.matches(sealed));
        assertFalse(pointer.matches(record("FAIL",
                "0000000000000000000000000000000000000000000000000000000000000000")));
    }

    /**
     * The decoder is pointed at whatever is in frame, so anything that is not ours is an
     * ordinary outcome and has to come back null rather than throw.
     */
    @Test
    public void rejectsPayloadsThatAreNotOurs() {
        assertNull(ReportCode.decode(null));
        assertNull(ReportCode.decode(""));
        assertNull(ReportCode.decode("CHASSIS-8821-B"));
        assertNull(ReportCode.decode("QCR"));
        assertNull(ReportCode.decode("QCRnot-base-32-at-all!!"));
        // Right prefix and right shape, one character corrupted into a different length.
        assertNull(ReportCode.decode(ReportCode.encode(record("PASS", DIGEST)).substring(0, 20)));
    }

    @Test
    public void payloadStaysInsideQrAlphanumericMode() {
        String body = ReportCode.encode(record("FAIL", DIGEST)).substring(ReportCode.PREFIX.length());
        for (char c : body.toCharArray()) {
            // Base32's alphabet is what lets the symbol use alphanumeric mode, two characters
            // per 11 bits, instead of falling back to byte mode.
            assertTrue("unexpected character " + c,
                    (c >= 'A' && c <= 'Z') || (c >= '2' && c <= '7'));
        }
    }

    // ── Base32 ──────────────────────────────────────────────────────────────

    @Test
    public void base32RoundTripsArbitraryBytes() {
        Random random = new Random(20260820L);
        for (int trial = 0; trial < 200; trial++) {
            byte[] data = new byte[ReportCode.PAYLOAD_BYTES];
            random.nextBytes(data);
            assertArrayEquals(data, ReportCode.base32Decode(ReportCode.base32Encode(data)));
        }
    }

    /** Hand transcription off a damaged label will not preserve case; the alphabet has none. */
    @Test
    public void base32DecodesLowercase() {
        String encoded = ReportCode.encode(record("PASS", DIGEST));
        ReportCode.Pointer pointer = ReportCode.decode(
                ReportCode.PREFIX + encoded.substring(ReportCode.PREFIX.length()).toLowerCase());

        assertNotNull(pointer);
        assertEquals(UUID_TEXT, pointer.uuid);
    }

    @Test
    public void packedPayloadIsTwentySixBytes() {
        assertEquals(ReportCode.PAYLOAD_BYTES,
                ReportCode.pack(UUID_TEXT, ReportCode.VERDICT_FAIL, DIGEST).length);
    }

    // ── Rendered symbol ─────────────────────────────────────────────────────

    /**
     * The one that would actually be noticed if it broke: what comes off the page has to read
     * back as what went on it.
     */
    @Test
    public void renderedQrDecodesBackToThePayload() throws Exception {
        String payload = ReportCode.encode(record("FAIL", DIGEST));
        Bitmap bitmap = ReportCode.toQrBitmap(payload, 360);

        assertEquals(payload, readQr(bitmap));
        ReportCode.Pointer pointer = ReportCode.decode(readQr(bitmap));
        assertNotNull(pointer);
        assertEquals(UUID_TEXT, pointer.uuid);
    }

    /**
     * Every uuid the app will ever mint has to fit the same symbol, so a sample of them all
     * decode rather than just the one convenient constant above.
     */
    @Test
    public void randomUuidsAllRoundTripThroughARenderedSymbol() throws Exception {
        for (int trial = 0; trial < 5; trial++) {
            InspectionRecord r = record("PASS", DIGEST);
            r.setUuid(UUID.randomUUID().toString());

            String payload = ReportCode.encode(r);
            ReportCode.Pointer pointer = ReportCode.decode(readQr(ReportCode.toQrBitmap(payload, 360)));

            assertNotNull(pointer);
            assertEquals(r.getUuid(), pointer.uuid);
        }
    }

    private static String readQr(Bitmap bitmap) throws Exception {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        Map<DecodeHintType, Object> hints = new EnumMap<>(DecodeHintType.class);
        hints.put(DecodeHintType.PURE_BARCODE, Boolean.TRUE);
        Result result = new MultiFormatReader().decode(
                new BinaryBitmap(new HybridBinarizer(
                        new RGBLuminanceSource(width, height, pixels))),
                hints);
        return result.getText();
    }
}
