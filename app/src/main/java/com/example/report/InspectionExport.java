package com.example.report;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.SlotResult;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.TimeZone;

/**
 * Turns a filed inspection into the two things the PDF cannot be: a digest that makes a later
 * edit detectable, and a machine-readable file that leaves the device without being retyped.
 *
 * <p>The PDF is for a person. It is also the only artefact this app has ever produced, which
 * means an SME wanting the numbers in a spreadsheet or an ERP has had to read them off a page
 * and type them back in. JSON carries the whole record including the per-slot findings; CSV
 * carries one row per inspection, because that is what actually gets opened.
 *
 * <p>Serialisation is hand-rolled over {@code org.json} rather than through Gson or Moshi. The
 * app already writes five files this way ({@code FeatureLayout}, {@code ReferenceLibrary},
 * {@code ThresholdTuner}, {@code FeatureTypeRegistry}), it is offline with no reflection-based
 * dependency anywhere, and the schema here is three flat tables.
 */
public final class InspectionExport {

    /**
     * Version of the export schema, stamped into every record sealed by this build.
     *
     * <p>Bump this when the canonical form changes shape, because that changes every digest it
     * produces: a reader has to know which rules a stored digest was computed under before it
     * can decide whether a mismatch means tampering or just an older writer.
     */
    public static final int SCHEMA_VERSION = 1;

    private InspectionExport() { }

    // ── Canonical form and digest ───────────────────────────────────────────

    /**
     * The exact byte sequence a digest is taken over.
     *
     * <p>Two properties are load-bearing and neither is free:
     *
     * <ul>
     *   <li><b>Unambiguous.</b> Every value is written length-prefixed as
     *       {@code name:length:value}, so no arrangement of separators or newlines inside a
     *       part id or an operator note can be made to look like a field boundary. Escaping
     *       would work too; counting bytes is simply harder to get wrong.
     *   <li><b>Order-independent.</b> Child rows are sorted before they are written, because
     *       the DAO returns them in whatever order SQLite chose. Without this, re-running a
     *       verification after a VACUUM could report tampering on an untouched record.
     * </ul>
     *
     * <p>The rowid is excluded deliberately: it is device-local and gets reassigned by an
     * export/import round trip, which would make every restored record fail its own seal. The
     * uuid <em>is</em> included, so a digest binds content to identity and two records cannot
     * have their identities swapped without detection.
     *
     * <p>A slot's size ratio is written only when it is present. {@link #verify} recomputes the
     * form under the current rules, so a field that every older record wrote as null would
     * change the bytes of every seal already filed and mark all of them tampered. Writing it
     * conditionally keeps those forms byte-identical while still sealing the value on records
     * that have one: nulling a stored ratio removes its field and breaks the digest, and adding
     * one to an old record inserts a field and does the same.
     */
    @NonNull
    public static String canonicalForm(@NonNull InspectionRecord record,
                                       @Nullable List<FeatureResult> features,
                                       @Nullable List<SlotResult> slots) {
        StringBuilder sb = new StringBuilder();
        field(sb, "schema", String.valueOf(SCHEMA_VERSION));
        field(sb, "uuid", record.getUuid());
        field(sb, "partId", record.getPartId());
        field(sb, "canonicalPartId", record.getCanonicalPartId());
        field(sb, "timestamp", String.valueOf(record.getTimestamp()));
        field(sb, "verdict", record.getPassOrFail());
        field(sb, "operator", record.getOperator());
        field(sb, "imageUri", record.getImageUri());
        field(sb, "notes", record.getNotes());
        field(sb, "supersedes", record.getSupersedesUuid());

        for (FeatureResult f : sortedFeatures(features)) {
            field(sb, "feature", f.getFeatureType());
            field(sb, "expected", String.valueOf(f.getExpectedCount()));
            field(sb, "detected", String.valueOf(f.getDetectedCount()));
            field(sb, "featureStatus", f.getStatus());
            field(sb, "severity", String.valueOf(f.getSeverity()));
        }

        for (SlotResult s : sortedSlots(slots)) {
            field(sb, "slot", String.valueOf(s.getSlotId()));
            field(sb, "slotExpected", s.getExpectedType());
            field(sb, "slotFound", s.getFoundType());
            field(sb, "slotStatus", s.getStatus());
            field(sb, "slotRelX", coord(s.getRelX()));
            field(sb, "slotRelY", coord(s.getRelY()));
            field(sb, "slotOffset", s.getOffsetPct() == null ? null : coord(s.getOffsetPct()));
            if (s.getSizeRatio() != null) field(sb, "slotSize", coord(s.getSizeRatio()));
        }
        return sb.toString();
    }

    /** Lowercase hex SHA-256 of {@link #canonicalForm}. */
    @NonNull
    public static String digest(@NonNull InspectionRecord record,
                                @Nullable List<FeatureResult> features,
                                @Nullable List<SlotResult> slots) {
        return sha256Hex(canonicalForm(record, features, slots));
    }

    /**
     * Whether a record still matches the seal it was filed with.
     *
     * <p>False for an unsealed record too. That is not a claim it was tampered with — rows
     * filed before sealing existed have nothing to check against — so callers have to
     * distinguish the two by looking at {@link InspectionRecord#getDigest()} themselves rather
     * than reading a false here as evidence of anything.
     */
    public static boolean verify(@NonNull InspectionRecord record,
                                 @Nullable List<FeatureResult> features,
                                 @Nullable List<SlotResult> slots) {
        String stored = record.getDigest();
        return stored != null && stored.equals(digest(record, features, slots));
    }

    @NonNull
    static String sha256Hex(@NonNull String content) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(content.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder(hash.length * 2);
            for (byte b : hash) {
                hex.append(Character.forDigit((b >> 4) & 0xF, 16));
                hex.append(Character.forDigit(b & 0xF, 16));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException e) {
            // Every Android release ships SHA-256; there is no sane fallback, and quietly
            // returning a weaker or empty digest would be worse than failing here.
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }

    // ── JSON ────────────────────────────────────────────────────────────────

    /** The full record — header, per-type counts and per-position findings — as JSON. */
    @NonNull
    public static String toJson(@NonNull InspectionRecord record,
                                @Nullable List<FeatureResult> features,
                                @Nullable List<SlotResult> slots) throws JSONException {
        JSONObject root = new JSONObject();
        root.put("schemaVersion", SCHEMA_VERSION);
        root.put("uuid", nullable(record.getUuid()));
        root.put("partId", nullable(record.getPartId()));
        root.put("canonicalPartId", nullable(record.getCanonicalPartId()));
        root.put("timestamp", record.getTimestamp());
        root.put("timestampIso", iso8601(record.getTimestamp()));
        root.put("verdict", nullable(record.getPassOrFail()));
        root.put("operator", nullable(record.getOperator()));
        root.put("notes", nullable(record.getNotes()));
        root.put("imageUri", nullable(record.getImageUri()));
        root.put("supersedesUuid", nullable(record.getSupersedesUuid()));
        root.put("digest", nullable(record.getDigest()));
        // Stated rather than implied: a reader should not have to infer from a null digest
        // that this record predates sealing and carries no integrity guarantee.
        root.put("sealed", record.getDigest() != null);

        JSONArray featureArray = new JSONArray();
        for (FeatureResult f : sortedFeatures(features)) {
            JSONObject o = new JSONObject();
            o.put("featureType", nullable(f.getFeatureType()));
            o.put("expectedCount", f.getExpectedCount());
            o.put("detectedCount", f.getDetectedCount());
            o.put("status", nullable(f.getStatus()));
            o.put("severity", f.getSeverity());
            featureArray.put(o);
        }
        root.put("featureResults", featureArray);

        JSONArray slotArray = new JSONArray();
        for (SlotResult s : sortedSlots(slots)) {
            JSONObject o = new JSONObject();
            o.put("slot", s.getSlotId());
            o.put("expectedType", nullable(s.getExpectedType()));
            o.put("foundType", nullable(s.getFoundType()));
            o.put("status", nullable(s.getStatus()));
            o.put("relX", round(s.getRelX()));
            o.put("relY", round(s.getRelY()));
            o.put("zone", nullable(s.getZone()));
            o.put("offsetPct", s.getOffsetPct() == null
                    ? JSONObject.NULL : round(s.getOffsetPct()));
            // Detected size over the reference's, corrected for framing; null when not judged.
            o.put("sizeRatio", s.getSizeRatio() == null
                    ? JSONObject.NULL : round(s.getSizeRatio()));
            slotArray.put(o);
        }
        root.put("slotResults", slotArray);

        return root.toString(2);
    }

    // ── CSV ─────────────────────────────────────────────────────────────────

    /** Header row for {@link #toCsvRow}. */
    @NonNull
    public static String csvHeader() {
        return "uuid,partId,timestampIso,verdict,operator,sealed,"
                + "criticalTypes,expectedTotal,detectedTotal,"
                + "slotsChecked,slotsMissing,slotsWrongType,slotsDisplaced,notes";
    }

    /**
     * One inspection as a CSV row — the roll-up view, not the full record.
     *
     * <p>Per-slot detail is deliberately not flattened in here. A row whose width depends on
     * how many slots the part happens to have is not a spreadsheet column, and the JSON
     * already carries that detail for anything that needs it.
     */
    @NonNull
    public static String toCsvRow(@NonNull InspectionRecord record,
                                  @Nullable List<FeatureResult> features,
                                  @Nullable List<SlotResult> slots) {
        int types = 0, expected = 0, detected = 0;
        for (FeatureResult f : sortedFeatures(features)) {
            types++;
            expected += f.getExpectedCount();
            detected += f.getDetectedCount();
        }
        int checked = 0, missing = 0, wrong = 0, displaced = 0;
        for (SlotResult s : sortedSlots(slots)) {
            checked++;
            if ("MISSING".equals(s.getStatus())) missing++;
            else if ("WRONG_TYPE".equals(s.getStatus())) wrong++;
            else if ("DISPLACED".equals(s.getStatus())) displaced++;
        }

        StringBuilder sb = new StringBuilder();
        csv(sb, record.getUuid());
        csv(sb, record.getPartId());
        csv(sb, iso8601(record.getTimestamp()));
        csv(sb, record.getPassOrFail());
        csv(sb, record.getOperator());
        csv(sb, String.valueOf(record.getDigest() != null));
        csv(sb, String.valueOf(types));
        csv(sb, String.valueOf(expected));
        csv(sb, String.valueOf(detected));
        csv(sb, String.valueOf(checked));
        csv(sb, String.valueOf(missing));
        csv(sb, String.valueOf(wrong));
        csv(sb, String.valueOf(displaced));
        sb.append(csvCell(record.getNotes()));
        return sb.toString();
    }

    // ── Plumbing ────────────────────────────────────────────────────────────

    private static void field(StringBuilder sb, String name, @Nullable String value) {
        String v = value == null ? "" : value;
        sb.append(name).append(':')
                .append(v.getBytes(StandardCharsets.UTF_8).length).append(':')
                .append(v).append('\n');
    }

    /**
     * Floats are written through a fixed-precision decimal rather than {@code toString}.
     *
     * <p>A digest over a raw float rendering is a digest over the platform's float formatter,
     * which is not something to bet a tamper check on. Six places is well inside what the
     * report ever displays for a normalised coordinate.
     */
    private static String coord(float value) {
        return String.format(Locale.ROOT, "%.6f", value);
    }

    private static double round(float value) {
        return Math.round(value * 1_000_000d) / 1_000_000d;
    }

    private static Object nullable(@Nullable String value) {
        return value == null ? JSONObject.NULL : value;
    }

    private static String iso8601(long millis) {
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.ROOT);
        // UTC, not the device zone: an exported file gets opened somewhere else, and a
        // timestamp whose offset is unstated is a timestamp that cannot be compared.
        format.setTimeZone(TimeZone.getTimeZone("UTC"));
        return format.format(new Date(millis));
    }

    private static void csv(StringBuilder sb, @Nullable String value) {
        sb.append(csvCell(value)).append(',');
    }

    /** RFC 4180 quoting: always quote, and double any quote inside. */
    private static String csvCell(@Nullable String value) {
        String v = value == null ? "" : value;
        return "\"" + v.replace("\"", "\"\"") + "\"";
    }

    private static List<FeatureResult> sortedFeatures(@Nullable List<FeatureResult> features) {
        List<FeatureResult> copy = features == null ? new ArrayList<>() : new ArrayList<>(features);
        Collections.sort(copy, new Comparator<FeatureResult>() {
            @Override
            public int compare(FeatureResult a, FeatureResult b) {
                return nullSafe(a.getFeatureType()).compareTo(nullSafe(b.getFeatureType()));
            }
        });
        return copy;
    }

    private static List<SlotResult> sortedSlots(@Nullable List<SlotResult> slots) {
        List<SlotResult> copy = slots == null ? new ArrayList<>() : new ArrayList<>(slots);
        Collections.sort(copy, new Comparator<SlotResult>() {
            @Override
            public int compare(SlotResult a, SlotResult b) {
                int bySlot = Integer.compare(a.getSlotId(), b.getSlotId());
                // Slot numbers are unique within a layout, so the tie-break only matters for
                // malformed data — but a comparator returning 0 for distinct rows would make
                // the sort order, and therefore the digest, depend on the input order.
                return bySlot != 0 ? bySlot
                        : nullSafe(a.getExpectedType()).compareTo(nullSafe(b.getExpectedType()));
            }
        });
        return copy;
    }

    private static String nullSafe(@Nullable String value) {
        return value == null ? "" : value;
    }
}
