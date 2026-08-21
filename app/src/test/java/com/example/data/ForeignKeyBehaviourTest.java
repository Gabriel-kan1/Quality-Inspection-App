package com.example.data;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import android.content.Context;
import android.database.Cursor;

import androidx.room.Room;
import androidx.test.core.app.ApplicationProvider;

import com.example.data.entity.ClassificationEvent;
import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.SlotResult;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.util.Collections;
import java.util.List;

/**
 * Covers what the foreign keys actually do at runtime, against a real Room instance rather than
 * hand-written SQL.
 *
 * <p>Declaring a constraint and enforcing one are different things — Room turns on
 * {@code PRAGMA foreign_keys} itself, and a schema that declares constraints nobody enforces is
 * the worst of both outcomes: the annotations read as a guarantee and the database gives none.
 * So the first thing proved here is that an invalid insert is actually rejected.
 */
@RunWith(RobolectricTestRunner.class)
public class ForeignKeyBehaviourTest {

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

    private long insertInspection(String partId) {
        InspectionRecord record = new InspectionRecord();
        record.setPartId(partId);
        record.setCanonicalPartId(partId);
        record.setUuid(java.util.UUID.randomUUID().toString());
        record.setTimestamp(1_700_000_000_000L);
        record.setPassOrFail("PASS");
        return db.inspectionDao().insertInspection(record);
    }

    private ClassificationEvent event(String partId, boolean reviewed) {
        ClassificationEvent e = new ClassificationEvent();
        e.setPartId(partId);
        e.setTimestamp(1_700_000_000_000L);
        e.setCropPath("/crops/a.png");
        e.setPredictedType("Bolt");
        e.setPredictedScore(0.81f);
        e.setReviewed(reviewed);
        e.setConfirmedType("Bolt");
        e.setEncoderId("dinov2");
        return e;
    }

    // ── Enforcement ─────────────────────────────────────────────────────────

    @Test
    public void childRowsPointingAtNothingAreRejected() {
        try {
            db.inspectionDao().insertFeatureResults(Collections.singletonList(
                    new FeatureResult(4040, "Bolt", 4, 3, "MISSING", 2)));
            fail("a FeatureResult was accepted with no inspection behind it");
        } catch (RuntimeException expected) {
            // The constraint is live, which is the whole point of declaring it.
        }
    }

    /**
     * Unassigned is a normal state, not a violation: events are logged during the pipeline,
     * before an inspection row exists, and are attached later — or never, if the operator does
     * not save the report.
     */
    @Test
    public void eventsWithNoInspectionAreAccepted() {
        long id = db.classificationEventDao().insert(event("CHASSIS-8821-B", false));

        ClassificationEvent stored = db.classificationEventDao().getById(id);
        assertNotNull(stored);
        assertNull(stored.getInspectionId());
    }

    // ── Cascade: the report goes with the inspection ────────────────────────

    @Test
    public void deletingAnInspectionTakesItsReportRowsWithIt() {
        long id = insertInspection("CHASSIS-8821-B");
        db.inspectionDao().insertFeatureResults(Collections.singletonList(
                new FeatureResult(id, "Bolt", 4, 3, "MISSING", 2)));
        db.inspectionDao().insertSlotResults(Collections.singletonList(
                new SlotResult(id, 1, "Bolt", null, "MISSING",
                        0.5f, 0.5f, 0.25f, 0.75f, null, "upper right")));

        assertEquals(1, db.inspectionDao().getFeatureResultsForInspection(id).size());
        assertEquals(1, db.inspectionDao().getSlotResultsForInspection(id).size());

        db.inspectionDao().deleteInspection(id);

        // One call, not three. The two hand-rolled child deletes that used to sit beside this
        // in the DAO are gone; the schema does the work now.
        assertEquals(0, db.inspectionDao().getFeatureResultsForInspection(id).size());
        assertEquals(0, db.inspectionDao().getSlotResultsForInspection(id).size());
    }

    @Test
    public void deletingOneInspectionLeavesAnotherIntact() {
        long kept = insertInspection("CHASSIS-8821-B");
        long removed = insertInspection("ENGINE-COV-X42");
        db.inspectionDao().insertFeatureResults(Collections.singletonList(
                new FeatureResult(kept, "Bolt", 4, 4, "MATCH", 2)));
        db.inspectionDao().insertFeatureResults(Collections.singletonList(
                new FeatureResult(removed, "Rivet", 2, 1, "MISSING", 2)));

        db.inspectionDao().deleteInspection(removed);

        assertEquals(1, db.inspectionDao().getFeatureResultsForInspection(kept).size());
        assertEquals(0, db.inspectionDao().getFeatureResultsForInspection(removed).size());
    }

    // ── Set null: the training data outlives the report ─────────────────────

    /**
     * The distinction this whole change turns on. A reviewed crop is labelled training data;
     * deleting a report must not cost the classifier its history for that part.
     */
    @Test
    public void deletingAnInspectionDetachesItsEventsButKeepsThem() {
        long id = insertInspection("CHASSIS-8821-B");
        long eventId = db.classificationEventDao().insert(event("CHASSIS-8821-B", true));
        db.classificationEventDao().assignInspection(id, Collections.singletonList(eventId));

        assertEquals(Long.valueOf(id), db.classificationEventDao().getById(eventId).getInspectionId());

        db.inspectionDao().deleteInspection(id);

        ClassificationEvent survivor = db.classificationEventDao().getById(eventId);
        assertNotNull("the labelled crop was deleted with the report", survivor);
        assertNull(survivor.getInspectionId());
        assertTrue("the human label was lost", survivor.isReviewed());
    }

    @Test
    public void theTunerStillSeesEventsWhoseReportWasDeleted() {
        long id = insertInspection("CHASSIS-8821-B");
        long eventId = db.classificationEventDao().insert(event("CHASSIS-8821-B", true));
        db.classificationEventDao().assignInspection(id, Collections.singletonList(eventId));

        db.inspectionDao().deleteInspection(id);

        // ThresholdTuner reads by partId and reviewed, never by inspectionId, so report
        // housekeeping has to be invisible to it.
        List<ClassificationEvent> forTuning =
                db.classificationEventDao().getRecentReviewedForPart("CHASSIS-8821-B", 50);
        assertEquals(1, forTuning.size());
        assertEquals(1, db.classificationEventDao().countReviewedForPart("CHASSIS-8821-B"));
    }

    // ── Indices ─────────────────────────────────────────────────────────────

    /**
     * Checked against the schema Room builds from the annotations, not the one the migration
     * writes — those are two separate code paths and a fresh install only ever sees this one.
     */
    @Test
    public void roomCreatesTheIndicesOnAFreshInstall() {
        assertTrue(hasIndexOn("FeatureResult", "inspectionId"));
        assertTrue(hasIndexOn("SlotResult", "inspectionId"));
        assertTrue(hasIndexOn("ClassificationEvent", "inspectionId"));
        assertTrue(hasIndexOn("ClassificationEvent", "partId"));
        assertTrue(hasIndexOn("InspectionRecord", "partId"));
    }

    /** True when any index on {@code table} has {@code column} as its first column. */
    private boolean hasIndexOn(String table, String column) {
        try (Cursor list = db.getOpenHelper().getReadableDatabase()
                .query("PRAGMA index_list(" + table + ")")) {
            while (list.moveToNext()) {
                String indexName = list.getString(list.getColumnIndex("name"));
                try (Cursor info = db.getOpenHelper().getReadableDatabase()
                        .query("PRAGMA index_info(" + indexName + ")")) {
                    if (info.moveToNext()
                            && column.equals(info.getString(info.getColumnIndex("name")))) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
}
