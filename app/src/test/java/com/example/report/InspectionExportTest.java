package com.example.report;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.SlotResult;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Covers the seal and the machine-readable exports.
 *
 * <p>The digest is the only thing in the app making a claim about whether a filed record has
 * been altered, so the tests that matter are the ones that would let it lie: a change nobody
 * notices, or a false alarm on an untouched record. Both are here.
 *
 * <p>Robolectric rather than plain JUnit only because {@code org.json} is a stub in the unit
 * test android.jar. The canonical form and the CSV need nothing from the framework.
 */
@RunWith(RobolectricTestRunner.class)
public class InspectionExportTest {

    private static InspectionRecord record() {
        InspectionRecord r = new InspectionRecord();
        r.setId(7);
        r.setUuid("3f2504e0-4f89-41d3-9a0c-0305e82c3301");
        r.setPartId("CHASSIS-8821-B");
        r.setCanonicalPartId("CHASSIS-8821-B");
        r.setTimestamp(1_700_000_000_000L);
        r.setPassOrFail("FAIL");
        r.setOperator("SHIFT-A");
        r.setNotes("Rework raised.");
        r.setImageUri("file:///data/COMP_CHASSIS_1.jpg");
        return r;
    }

    private static FeatureResult feature(String type, int expected, int detected, String status) {
        return new FeatureResult(7, type, expected, detected, status, 2);
    }

    private static SlotResult slot(int id, String status) {
        return new SlotResult(7, id, "Bolt", "PRESENT".equals(status) ? "Bolt" : null,
                status, 0.5f, 0.5f, 0.25f, 0.75f, null, "upper right");
    }

    private static List<FeatureResult> features() {
        return new ArrayList<>(Arrays.asList(
                feature("Bolt", 4, 3, "MISSING"),
                feature("Rivet", 2, 2, "MATCH")));
    }

    private static List<SlotResult> slots() {
        return new ArrayList<>(Arrays.asList(
                slot(1, "PRESENT"), slot(2, "MISSING"), slot(3, "PRESENT")));
    }

    // ── Canonical form ──────────────────────────────────────────────────────

    @Test
    public void canonicalFormIsStableAcrossRuns() {
        assertEquals(InspectionExport.canonicalForm(record(), features(), slots()),
                InspectionExport.canonicalForm(record(), features(), slots()));
    }

    /**
     * The DAO hands back child rows in whatever order SQLite chose. If that order reached the
     * digest, a VACUUM or a schema change could make an untouched record fail its own seal.
     */
    @Test
    public void childRowOrderDoesNotChangeTheDigest() {
        List<FeatureResult> shuffledFeatures = features();
        Collections.reverse(shuffledFeatures);
        List<SlotResult> shuffledSlots = slots();
        Collections.reverse(shuffledSlots);

        assertEquals(InspectionExport.digest(record(), features(), slots()),
                InspectionExport.digest(record(), shuffledFeatures, shuffledSlots));
    }

    /**
     * The rowid is device-local and gets reassigned by an export/import round trip. If it were
     * in the digest, every restored record would report as tampered with.
     */
    @Test
    public void rowIdDoesNotChangeTheDigest() {
        InspectionRecord renumbered = record();
        renumbered.setId(4242);
        assertEquals(InspectionExport.digest(record(), features(), slots()),
                InspectionExport.digest(renumbered, features(), slots()));
    }

    /**
     * Length prefixing exists so no arrangement of separators inside a free-text field can be
     * made to look like a field boundary. Without it, moving characters between two adjacent
     * fields would leave the serialised form — and so the digest — unchanged.
     */
    @Test
    public void contentMovedBetweenAdjacentFieldsChangesTheDigest() {
        InspectionRecord a = record();
        a.setPartId("AB");
        a.setCanonicalPartId("C");

        InspectionRecord b = record();
        b.setPartId("A");
        b.setCanonicalPartId("BC");

        assertNotEquals(InspectionExport.digest(a, features(), slots()),
                InspectionExport.digest(b, features(), slots()));
    }

    @Test
    public void everyMeaningfulFieldChangeIsDetected() {
        String baseline = InspectionExport.digest(record(), features(), slots());

        InspectionRecord verdictFlipped = record();
        verdictFlipped.setPassOrFail("PASS");
        assertNotEquals("verdict", baseline,
                InspectionExport.digest(verdictFlipped, features(), slots()));

        InspectionRecord operatorChanged = record();
        operatorChanged.setOperator("SHIFT-B");
        assertNotEquals("operator", baseline,
                InspectionExport.digest(operatorChanged, features(), slots()));

        InspectionRecord notesChanged = record();
        notesChanged.setNotes("Rework raised!");
        assertNotEquals("notes", baseline,
                InspectionExport.digest(notesChanged, features(), slots()));

        List<FeatureResult> countChanged = features();
        countChanged.get(0).setDetectedCount(4);
        assertNotEquals("detected count", baseline,
                InspectionExport.digest(record(), countChanged, slots()));

        List<SlotResult> statusChanged = slots();
        statusChanged.get(1).setStatus("PRESENT");
        assertNotEquals("slot status", baseline,
                InspectionExport.digest(record(), features(), statusChanged));

        List<SlotResult> slotDropped = slots();
        slotDropped.remove(1);
        assertNotEquals("dropped slot", baseline,
                InspectionExport.digest(record(), features(), slotDropped));
    }

    // ── Verification ────────────────────────────────────────────────────────

    @Test
    public void sealedRecordVerifies() {
        InspectionRecord sealed = record();
        sealed.setDigest(InspectionExport.digest(sealed, features(), slots()));
        assertTrue(InspectionExport.verify(sealed, features(), slots()));
    }

    @Test
    public void tamperedRecordFailsVerification() {
        InspectionRecord sealed = record();
        sealed.setDigest(InspectionExport.digest(sealed, features(), slots()));
        sealed.setPassOrFail("PASS");
        assertFalse(InspectionExport.verify(sealed, features(), slots()));
    }

    /**
     * An unsealed record is not a tampered one. It has to fail here — there is nothing to
     * check — but the report has to describe it as unsealed rather than as altered, which is
     * why the null digest stays visible on the record instead of being papered over.
     */
    @Test
    public void unsealedRecordFailsVerificationButKeepsItsNullDigest() {
        InspectionRecord legacy = record();
        legacy.setDigest(null);
        assertFalse(InspectionExport.verify(legacy, features(), slots()));
        org.junit.Assert.assertNull(legacy.getDigest());
    }

    // ── JSON ────────────────────────────────────────────────────────────────

    @Test
    public void jsonCarriesTheWholeRecord() throws Exception {
        InspectionRecord sealed = record();
        sealed.setDigest(InspectionExport.digest(sealed, features(), slots()));

        JSONObject json = new JSONObject(InspectionExport.toJson(sealed, features(), slots()));
        assertEquals(InspectionExport.SCHEMA_VERSION, json.getInt("schemaVersion"));
        assertEquals("3f2504e0-4f89-41d3-9a0c-0305e82c3301", json.getString("uuid"));
        assertEquals("CHASSIS-8821-B", json.getString("partId"));
        assertEquals("FAIL", json.getString("verdict"));
        assertEquals("SHIFT-A", json.getString("operator"));
        assertTrue(json.getBoolean("sealed"));
        // UTC, so an exported file means the same thing wherever it is opened.
        assertEquals("2023-11-14T22:13:20Z", json.getString("timestampIso"));

        JSONArray features = json.getJSONArray("featureResults");
        assertEquals(2, features.length());
        assertEquals("Bolt", features.getJSONObject(0).getString("featureType"));
        assertEquals(3, features.getJSONObject(0).getInt("detectedCount"));

        JSONArray slots = json.getJSONArray("slotResults");
        assertEquals(3, slots.length());
        assertEquals("MISSING", slots.getJSONObject(1).getString("status"));
    }

    @Test
    public void jsonMarksAnUnsealedRecordAsUnsealed() throws Exception {
        JSONObject json = new JSONObject(InspectionExport.toJson(record(), features(), slots()));
        assertFalse(json.getBoolean("sealed"));
        assertTrue(json.isNull("digest"));
    }

    // ── CSV ─────────────────────────────────────────────────────────────────

    @Test
    public void csvRollsUpTheCountsAndMatchesItsHeader() {
        String row = InspectionExport.toCsvRow(record(), features(), slots());
        assertEquals(InspectionExport.csvHeader().split(",", -1).length,
                row.split(",", -1).length);

        List<String> cells = Arrays.asList(row.split(",", -1));
        assertEquals("\"CHASSIS-8821-B\"", cells.get(1));
        assertEquals("\"FAIL\"", cells.get(3));
        assertEquals("\"SHIFT-A\"", cells.get(4));
        assertEquals("\"false\"", cells.get(5));
        // 2 critical types, 4+2 expected, 3+2 detected, 3 slots checked, 1 missing.
        assertEquals("\"2\"", cells.get(6));
        assertEquals("\"6\"", cells.get(7));
        assertEquals("\"5\"", cells.get(8));
        assertEquals("\"3\"", cells.get(9));
        assertEquals("\"1\"", cells.get(10));
    }

    /**
     * Notes are free text an operator typed, so they will eventually contain a comma, a quote
     * and a newline. A CSV that breaks on any of those shifts every later column of that row.
     */
    @Test
    public void csvQuotesFreeTextThatWouldOtherwiseBreakTheRow() {
        InspectionRecord awkward = record();
        awkward.setNotes("Bolt 3, \"loose\"\nre-torqued");
        String row = InspectionExport.toCsvRow(awkward, features(), slots());

        assertTrue(row.endsWith("\"Bolt 3, \"\"loose\"\"\nre-torqued\""));
    }

    @Test
    public void emptyInspectionStillProducesAWellFormedRow() {
        String row = InspectionExport.toCsvRow(record(), null, null);
        assertEquals(InspectionExport.csvHeader().split(",", -1).length,
                row.split(",", -1).length);
    }

    // ── Size ratio ──────────────────────────────────────────────────────────

    /**
     * A ratio is sealed when present and absent from the form when null. The absence matters:
     * every record filed before the column existed has null there, and its stored digest was
     * computed over a form with no such field — so the field must not appear for null, or
     * every one of those seals would now read as tampered.
     */
    @Test
    public void sizeRatioIsSealedOnlyWhenPresent() {
        String withoutField = InspectionExport.canonicalForm(record(), features(), slots());
        assertFalse(withoutField.contains("slotSize"));

        List<SlotResult> judged = slots();
        judged.get(0).setSizeRatio(1.42f);
        String withField = InspectionExport.canonicalForm(record(), features(), judged);
        assertTrue(withField.contains("slotSize:8:1.420000"));
        assertNotEquals(InspectionExport.digest(record(), features(), slots()),
                InspectionExport.digest(record(), features(), judged));

        InspectionRecord sealed = record();
        sealed.setDigest(InspectionExport.digest(sealed, features(), judged));
        assertTrue(InspectionExport.verify(sealed, features(), judged));
        judged.get(0).setSizeRatio(null);
        assertFalse("nulling a sealed ratio must break the seal",
                InspectionExport.verify(sealed, features(), judged));
    }

    @Test
    public void jsonCarriesTheSizeRatioAndItsAbsence() throws Exception {
        List<SlotResult> judged = slots();
        judged.get(0).setSizeRatio(1.42f);
        JSONObject root = new JSONObject(InspectionExport.toJson(record(), features(), judged));
        JSONArray slotArray = root.getJSONArray("slotResults");
        assertEquals(1.42, slotArray.getJSONObject(0).getDouble("sizeRatio"), 1e-6);
        assertTrue(slotArray.getJSONObject(1).isNull("sizeRatio"));
    }
}
