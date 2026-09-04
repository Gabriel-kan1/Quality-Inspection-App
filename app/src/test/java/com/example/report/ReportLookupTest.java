package com.example.report;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.graphics.Bitmap;

import androidx.room.Room;
import androidx.test.core.app.ApplicationProvider;

import com.example.data.AppDatabase;
import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.SlotResult;
import com.google.zxing.BinaryBitmap;
import com.google.zxing.MultiFormatReader;
import com.google.zxing.RGBLuminanceSource;
import com.google.zxing.common.HybridBinarizer;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.GraphicsMode;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * The whole point of the printed code, as one test: a rendered QR goes in, the filed inspection
 * comes out.
 *
 * <p>{@code ReportCodeTest} proves the payload round-trips and {@code ForeignKeyBehaviourTest}
 * proves the database behaves. Neither covers the join between them, which is where the feature
 * actually lives — and which was missing entirely until the scanner was wired up: the app could
 * print a pointer it had no way to follow.
 *
 * <p>Everything here except the camera is the real path {@code ReportScanFragment} takes.
 */
@RunWith(RobolectricTestRunner.class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
public class ReportLookupTest {

    private AppDatabase db;

    @Before
    public void setUp() {
        Context context = ApplicationProvider.getApplicationContext();
        db = Room.inMemoryDatabaseBuilder(context, AppDatabase.class)
                .allowMainThreadQueries()
                .build();
    }

    @After
    public void tearDown() {
        db.close();
    }

    /** Files an inspection the way {@code ResultsFragment} does, seal and all. */
    private InspectionRecord fileInspection(String partId, String verdict) {
        InspectionRecord record = new InspectionRecord();
        record.setPartId(partId);
        record.setCanonicalPartId(partId);
        record.setUuid(UUID.randomUUID().toString());
        record.setTimestamp(1_700_000_000_000L);
        record.setPassOrFail(verdict);
        record.setOperator("SHIFT-A");

        long id = db.inspectionDao().insertInspection(record);
        record.setId(id);

        List<FeatureResult> features = Arrays.asList(
                new FeatureResult(id, "Bolt", 6, 5, "MISSING", 2),
                new FeatureResult(id, "Rivet", 4, 4, "MATCH", 2));
        db.inspectionDao().insertFeatureResults(features);

        List<SlotResult> slots = Collections.singletonList(
                new SlotResult(id, 1, "Bolt", null, "MISSING",
                        0.5f, 0.5f, 0.25f, 0.75f, null, "upper right"));
        db.inspectionDao().insertSlotResults(slots);

        String digest = InspectionExport.digest(record, features, slots);
        db.inspectionDao().sealInspection(id, digest, InspectionExport.SCHEMA_VERSION);
        record.setDigest(digest);
        return record;
    }

    /** Renders the code and reads it back, standing in for the camera. */
    private static String scan(InspectionRecord record) throws Exception {
        Bitmap bitmap = ReportCode.toQrBitmap(ReportCode.encode(record), 360);
        int w = bitmap.getWidth(), h = bitmap.getHeight();
        int[] pixels = new int[w * h];
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h);
        return new MultiFormatReader().decode(new BinaryBitmap(new HybridBinarizer(
                new RGBLuminanceSource(w, h, pixels)))).getText();
    }

    // ── The loop ────────────────────────────────────────────────────────────

    @Test
    public void aScannedCodeResolvesToTheInspectionItWasPrintedFor() throws Exception {
        InspectionRecord filed = fileInspection("CHASSIS-8821-B", "FAIL");

        ReportCode.Pointer pointer = ReportCode.decode(scan(filed));
        assertNotNull("the rendered code did not read back", pointer);

        InspectionRecord found = db.inspectionDao().getInspectionByUuid(pointer.uuid);
        assertNotNull("the scanned uuid resolved to nothing", found);

        // The rowid is what ReportFragment is actually navigated with.
        assertEquals(filed.getId(), found.getId());
        assertEquals("CHASSIS-8821-B", found.getPartId());
        assertEquals("FAIL", found.getPassOrFail());
        assertEquals(2, db.inspectionDao().getFeatureResultsForInspection(found.getId()).size());
        assertEquals(1, db.inspectionDao().getSlotResultsForInspection(found.getId()).size());
    }

    @Test
    public void theVerdictIsReadableFromTheCodeAloneWithoutTheDatabase() throws Exception {
        InspectionRecord filed = fileInspection("CHASSIS-8821-B", "FAIL");
        ReportCode.Pointer pointer = ReportCode.decode(scan(filed));

        assertNotNull(pointer);
        assertEquals("FAIL", pointer.verdictText());
    }

    /**
     * The digest check doing its job on a sealed record — the case the on-device run could not
     * reach, because the only inspection there predated sealing and carried zeroed bytes.
     */
    @Test
    public void theCodeConfirmsThePrintoutMatchesTheStoredRecord() throws Exception {
        InspectionRecord filed = fileInspection("CHASSIS-8821-B", "FAIL");
        ReportCode.Pointer pointer = ReportCode.decode(scan(filed));

        assertNotNull(pointer);
        assertNotNull("a sealed record should put real digest bytes in the code",
                pointer.digestPrefix);
        assertTrue(pointer.matches(db.inspectionDao().getInspectionByUuid(pointer.uuid)));
    }

    /**
     * A record edited after filing no longer matches the code printed on its report, which is
     * the warning {@code ReportScanFragment} raises before opening it.
     */
    @Test
    public void aRecordAlteredAfterPrintingNoLongerMatchesItsCode() throws Exception {
        InspectionRecord filed = fileInspection("CHASSIS-8821-B", "FAIL");
        ReportCode.Pointer printed = ReportCode.decode(scan(filed));
        assertNotNull(printed);

        // Re-seal the row over changed contents, as an edit would.
        List<FeatureResult> tampered = Collections.singletonList(
                new FeatureResult(filed.getId(), "Bolt", 6, 6, "MATCH", 2));
        String newDigest = InspectionExport.digest(filed, tampered, Collections.emptyList());
        db.inspectionDao().sealInspection(filed.getId(), newDigest,
                InspectionExport.SCHEMA_VERSION);

        InspectionRecord reloaded = db.inspectionDao().getInspectionByUuid(printed.uuid);
        assertNotNull(reloaded);
        assertFalse("the printed code should no longer match the altered record",
                printed.matches(reloaded));
    }

    /**
     * The code identifies a record; it does not contain one. On a device that never filed this
     * inspection the lookup finds nothing, and the scanner reports exactly that rather than
     * treating it as a failure.
     */
    @Test
    public void aCodeFromAnotherDeviceResolvesToNothing() throws Exception {
        InspectionRecord elsewhere = new InspectionRecord();
        elsewhere.setUuid(UUID.randomUUID().toString());
        elsewhere.setPassOrFail("PASS");
        elsewhere.setPartId("CHASSIS-8821-B");

        ReportCode.Pointer pointer = ReportCode.decode(scan(elsewhere));
        assertNotNull(pointer);
        // The verdict still reads — that much travels with the paper.
        assertEquals("PASS", pointer.verdictText());
        assertNull(db.inspectionDao().getInspectionByUuid(pointer.uuid));
    }

    @Test
    public void twoInspectionsOfTheSamePartGetDistinctCodes() throws Exception {
        InspectionRecord first = fileInspection("CHASSIS-8821-B", "FAIL");
        InspectionRecord second = fileInspection("CHASSIS-8821-B", "PASS");

        ReportCode.Pointer a = ReportCode.decode(scan(first));
        ReportCode.Pointer b = ReportCode.decode(scan(second));
        assertNotNull(a);
        assertNotNull(b);

        // Same part, same day, different reports: the code has to tell them apart, which is the
        // reason it carries a uuid rather than the part id.
        assertFalse(a.uuid.equals(b.uuid));
        assertEquals(first.getId(), db.inspectionDao().getInspectionByUuid(a.uuid).getId());
        assertEquals(second.getId(), db.inspectionDao().getInspectionByUuid(b.uuid).getId());
    }
}
