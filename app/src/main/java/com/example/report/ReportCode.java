package com.example.report;

import android.graphics.Bitmap;
import android.graphics.Color;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.data.entity.InspectionRecord;
import com.google.zxing.BarcodeFormat;
import com.google.zxing.EncodeHintType;
import com.google.zxing.WriterException;
import com.google.zxing.common.BitMatrix;
import com.google.zxing.qrcode.QRCodeWriter;
import com.google.zxing.qrcode.decoder.ErrorCorrectionLevel;

import java.nio.ByteBuffer;
import java.util.EnumMap;
import java.util.Map;
import java.util.UUID;

/**
 * The pointer printed on a report as a QR code.
 *
 * <h2>Why a pointer and not the report</h2>
 *
 * <p>A QR can hold up to about 2,953 bytes, so a whole inspection would technically fit — a
 * twelve-slot layout packs into roughly 135 bytes. It is still the wrong thing to print. Slot
 * count is a property of the part, not of the report, and the layout code accepts up to 120
 * slots; a full summary would swing from a 30 mm symbol to a 55 mm one depending on which part
 * was inspected, taking scan reliability and the page layout with it. This payload is a fixed
 * {@value #PAYLOAD_BYTES} bytes for every inspection ever filed.
 *
 * <h2>Why QR and not a 1D barcode</h2>
 *
 * <p>Code 128 tops out around 48 characters before the label stops being usable, has no error
 * correction at all, and has to be presented roughly square-on. QR carries Reed–Solomon parity
 * recovering 7–30% of a damaged symbol and scans at any rotation, which is what a label on a
 * workshop floor actually needs.
 *
 * <h2>Encoding</h2>
 *
 * <p>The 26 raw bytes are Base32'd rather than stuffed into QR's byte mode. Base32's alphabet
 * (A–Z, 2–7) lies entirely inside QR's alphanumeric mode, which packs two characters per 11
 * bits — so 42 characters cost about 29 bytes of symbol capacity against the 26 that byte mode
 * would use, and in exchange the payload survives every scanner and charset-guessing heuristic
 * between here and the reader, and stays transcribable by hand when the label is too damaged
 * to scan.
 */
public final class ReportCode {

    /** Marks the payload as this app's, and says which layout the bytes are in. */
    public static final String PREFIX = "QCR";

    /** Bumped only if the byte layout below changes. Byte 0 of the payload. */
    public static final byte FORMAT_VERSION = 1;

    /** 1 version + 16 uuid + 1 verdict + 8 digest prefix. */
    public static final int PAYLOAD_BYTES = 26;

    /**
     * How much of the SHA-256 travels in the code.
     *
     * <p>Eight bytes is not a signature and is not meant to be — anyone able to rewrite the
     * database can recompute the digest. It is enough to catch a report paired with the wrong
     * record, or edited after filing by someone who did not think to regenerate the code, and
     * the full digest is in the JSON export and the database for anything stronger.
     */
    public static final int DIGEST_PREFIX_BYTES = 8;

    private static final String BASE32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

    private ReportCode() { }

    /** Verdict as one byte. Unknown text maps to {@link #VERDICT_UNKNOWN} rather than throwing. */
    public static final byte VERDICT_PASS = 0;
    public static final byte VERDICT_FAIL = 1;
    public static final byte VERDICT_UNVERIFIED = 2;
    public static final byte VERDICT_UNKNOWN = 3;

    /** What a scanned payload turns back into. */
    public static final class Pointer {
        public final int formatVersion;
        @NonNull public final String uuid;
        public final byte verdict;
        /** Lowercase hex of the digest prefix, or null when the record was never sealed. */
        @Nullable public final String digestPrefix;

        Pointer(int formatVersion, @NonNull String uuid, byte verdict,
                @Nullable String digestPrefix) {
            this.formatVersion = formatVersion;
            this.uuid = uuid;
            this.verdict = verdict;
            this.digestPrefix = digestPrefix;
        }

        /** The verdict as the string the record stores. */
        @NonNull
        public String verdictText() {
            switch (verdict) {
                case VERDICT_PASS: return "PASS";
                case VERDICT_FAIL: return "FAIL";
                case VERDICT_UNVERIFIED: return "UNVERIFIED";
                default: return "UNKNOWN";
            }
        }

        /** Whether this pointer agrees with a record's stored digest. */
        public boolean matches(@NonNull InspectionRecord record) {
            String stored = record.getDigest();
            if (digestPrefix == null || stored == null) return false;
            return stored.length() >= digestPrefix.length()
                    && stored.regionMatches(true, 0, digestPrefix, 0, digestPrefix.length());
        }
    }

    // ── Encode ──────────────────────────────────────────────────────────────

    /**
     * The scannable payload for a record.
     *
     * @throws IllegalArgumentException if the record has no uuid, which means it was never
     *     sealed by this build and there is nothing for a code to point at.
     */
    @NonNull
    public static String encode(@NonNull InspectionRecord record) {
        if (record.getUuid() == null) {
            throw new IllegalArgumentException("record has no uuid to encode");
        }
        return PREFIX + base32Encode(
                pack(record.getUuid(), verdictCode(record.getPassOrFail()), record.getDigest()));
    }

    static byte[] pack(@NonNull String uuid, byte verdict, @Nullable String digestHex) {
        UUID parsed = UUID.fromString(uuid);
        ByteBuffer buffer = ByteBuffer.allocate(PAYLOAD_BYTES);
        buffer.put(FORMAT_VERSION);
        buffer.putLong(parsed.getMostSignificantBits());
        buffer.putLong(parsed.getLeastSignificantBits());
        buffer.put(verdict);
        // An unsealed record leaves the digest field zeroed rather than omitting it, so the
        // payload stays a fixed width and the symbol a fixed size. decode() reads all-zero
        // back as "no digest", which is what an unsealed record honestly has.
        byte[] digest = new byte[DIGEST_PREFIX_BYTES];
        if (digestHex != null) {
            for (int i = 0; i < DIGEST_PREFIX_BYTES && (i * 2 + 1) < digestHex.length(); i++) {
                int hi = Character.digit(digestHex.charAt(i * 2), 16);
                int lo = Character.digit(digestHex.charAt(i * 2 + 1), 16);
                if (hi < 0 || lo < 0) break;
                digest[i] = (byte) ((hi << 4) | lo);
            }
        }
        buffer.put(digest);
        return buffer.array();
    }

    public static byte verdictCode(@Nullable String passOrFail) {
        if ("PASS".equals(passOrFail)) return VERDICT_PASS;
        if ("FAIL".equals(passOrFail)) return VERDICT_FAIL;
        if ("UNVERIFIED".equals(passOrFail)) return VERDICT_UNVERIFIED;
        return VERDICT_UNKNOWN;
    }

    // ── Decode ──────────────────────────────────────────────────────────────

    /**
     * Reads a scanned payload back. Returns null for anything that is not one of ours — a
     * wrong prefix, a bad length, an unknown format version or a corrupt Base32 body.
     *
     * <p>Null rather than an exception because the caller is a scanner pointed at whatever
     * happens to be in frame, and "that was some other barcode" is an ordinary outcome, not an
     * error.
     */
    @Nullable
    public static Pointer decode(@Nullable String payload) {
        if (payload == null || !payload.startsWith(PREFIX)) return null;
        byte[] bytes = base32Decode(payload.substring(PREFIX.length()));
        if (bytes == null || bytes.length != PAYLOAD_BYTES) return null;

        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        int version = buffer.get() & 0xFF;
        if (version != FORMAT_VERSION) return null;
        UUID uuid = new UUID(buffer.getLong(), buffer.getLong());
        byte verdict = buffer.get();

        StringBuilder hex = new StringBuilder(DIGEST_PREFIX_BYTES * 2);
        boolean anySet = false;
        for (int i = 0; i < DIGEST_PREFIX_BYTES; i++) {
            byte b = buffer.get();
            if (b != 0) anySet = true;
            hex.append(Character.forDigit((b >> 4) & 0xF, 16));
            hex.append(Character.forDigit(b & 0xF, 16));
        }
        return new Pointer(version, uuid.toString(), verdict,
                anySet ? hex.toString() : null);
    }

    // ── Rendering ───────────────────────────────────────────────────────────

    /**
     * Renders a payload as a black-on-white square bitmap.
     *
     * <p>Error correction is level M (~15% recoverable) rather than the L this payload could
     * get away with. At 26 bytes the symbol is small either way, so the parity is close to
     * free, and the thing being protected is a label that will be handled in a workshop.
     */
    @NonNull
    public static Bitmap toQrBitmap(@NonNull String payload, int sizePx) throws WriterException {
        Map<EncodeHintType, Object> hints = new EnumMap<>(EncodeHintType.class);
        hints.put(EncodeHintType.ERROR_CORRECTION, ErrorCorrectionLevel.M);
        // The default is 4 modules of quiet zone. Kept, not trimmed: without it a decoder has
        // nothing to separate the symbol from the surrounding page.
        hints.put(EncodeHintType.MARGIN, 2);
        hints.put(EncodeHintType.CHARACTER_SET, "ISO-8859-1");

        BitMatrix matrix = new QRCodeWriter().encode(payload, BarcodeFormat.QR_CODE,
                sizePx, sizePx, hints);
        int width = matrix.getWidth();
        int height = matrix.getHeight();
        int[] pixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            int offset = y * width;
            for (int x = 0; x < width; x++) {
                pixels[offset + x] = matrix.get(x, y) ? Color.BLACK : Color.WHITE;
            }
        }
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }

    // ── Base32 (RFC 4648, unpadded) ─────────────────────────────────────────

    @NonNull
    static String base32Encode(@NonNull byte[] data) {
        StringBuilder out = new StringBuilder((data.length * 8 + 4) / 5);
        int buffer = 0;
        int bitsLeft = 0;
        for (byte b : data) {
            buffer = (buffer << 8) | (b & 0xFF);
            bitsLeft += 8;
            while (bitsLeft >= 5) {
                out.append(BASE32.charAt((buffer >> (bitsLeft - 5)) & 0x1F));
                bitsLeft -= 5;
            }
        }
        if (bitsLeft > 0) {
            out.append(BASE32.charAt((buffer << (5 - bitsLeft)) & 0x1F));
        }
        return out.toString();
    }

    @Nullable
    static byte[] base32Decode(@NonNull String text) {
        int buffer = 0;
        int bitsLeft = 0;
        byte[] out = new byte[text.length() * 5 / 8];
        int written = 0;
        for (int i = 0; i < text.length(); i++) {
            // Uppercased on the way in: some scanners and every hand transcription will get
            // the case wrong, and the alphabet has no lowercase to be confused with.
            int value = BASE32.indexOf(Character.toUpperCase(text.charAt(i)));
            if (value < 0) return null;
            buffer = (buffer << 5) | value;
            bitsLeft += 5;
            if (bitsLeft >= 8) {
                if (written >= out.length) return null;
                out[written++] = (byte) ((buffer >> (bitsLeft - 8)) & 0xFF);
                bitsLeft -= 8;
            }
        }
        return written == out.length ? out : null;
    }
}
