package com.example.data;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.database.Cursor;

import androidx.sqlite.db.SupportSQLiteDatabase;
import androidx.sqlite.db.SupportSQLiteOpenHelper;
import androidx.sqlite.db.framework.FrameworkSQLiteOpenHelperFactory;
import androidx.test.core.app.ApplicationProvider;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/**
 * Covers the v10 → v11 upgrade, which adds the foreign keys the child tables never had.
 *
 * <p>The migration has to leave the database in a state that <em>satisfies</em> the constraints
 * it just declared. Room runs {@code PRAGMA foreign_key_check} after migrating, so a device
 * carrying real history that fails it does not open — which makes this the one migration where
 * getting the repair pass wrong is not a data-quality problem but a crash on launch.
 *
 * <p>Driven against raw SQLite for the same reason as {@link InspectionRecordMigrationTest}:
 * {@code MigrationTestHelper} needs exported schema JSON and this project builds with
 * {@code exportSchema = false}.
 */
@RunWith(RobolectricTestRunner.class)
public class ChildTableMigrationTest {

    private SupportSQLiteOpenHelper helper;
    private SupportSQLiteDatabase db;

    @Before
    public void setUp() {
        Context context = ApplicationProvider.getApplicationContext();
        helper = new FrameworkSQLiteOpenHelperFactory().create(
                SupportSQLiteOpenHelper.Configuration.builder(context)
                        .name(null)
                        .callback(new SupportSQLiteOpenHelper.Callback(10) {
                            @Override
                            public void onCreate(SupportSQLiteDatabase database) {
                                createV10Schema(database);
                            }

                            @Override
                            public void onUpgrade(SupportSQLiteDatabase database, int from, int to) {
                                // Nothing: this test drives the migration itself.
                            }
                        })
                        .build());
        db = helper.getWritableDatabase();
    }

    @After
    public void tearDown() {
        helper.close();
    }

    /** The four tables exactly as v10 left them — no constraints, no indices on inspectionId. */
    private static void createV10Schema(SupportSQLiteDatabase db) {
        db.execSQL("CREATE TABLE IF NOT EXISTS InspectionRecord ("
                + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                + "partId TEXT, timestamp INTEGER NOT NULL, passOrFail TEXT, imageUri TEXT,"
                + "notes TEXT, uuid TEXT, digest TEXT, schemaVersion INTEGER,"
                + "supersedesUuid TEXT, operator TEXT, canonicalPartId TEXT)");
        db.execSQL("CREATE TABLE IF NOT EXISTS FeatureResult ("
                + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                + "inspectionId INTEGER NOT NULL, featureType TEXT,"
                + "expectedCount INTEGER NOT NULL, detectedCount INTEGER NOT NULL,"
                + "status TEXT, severity INTEGER NOT NULL)");
        db.execSQL("CREATE TABLE IF NOT EXISTS SlotResult ("
                + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                + "inspectionId INTEGER NOT NULL, slotId INTEGER NOT NULL, expectedType TEXT,"
                + "foundType TEXT, status TEXT, x REAL NOT NULL, y REAL NOT NULL,"
                + "relX REAL NOT NULL, relY REAL NOT NULL, zone TEXT, offsetPct REAL)");
        db.execSQL("CREATE TABLE IF NOT EXISTS ClassificationEvent ("
                + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                + "partId TEXT, inspectionId INTEGER NOT NULL, timestamp INTEGER NOT NULL,"
                + "cropPath TEXT, predictedType TEXT, predictedScore REAL NOT NULL,"
                + "runnerUpType TEXT, runnerUpScore REAL NOT NULL, reason TEXT,"
                + "similarityThreshold REAL NOT NULL, marginThreshold REAL NOT NULL,"
                + "encoderId TEXT, confirmedType TEXT, reviewed INTEGER NOT NULL,"
                + "reviewSource TEXT)");
    }

    private void insertInspection(long id) {
        db.execSQL("INSERT INTO InspectionRecord (id, partId, timestamp, passOrFail, uuid)"
                + " VALUES (?, 'CHASSIS-8821-B', 1700000000000, 'PASS', ?)",
                new Object[]{id, "uuid-" + id});
    }

    private void insertFeatureResult(long inspectionId, String type) {
        db.execSQL("INSERT INTO FeatureResult"
                + " (inspectionId, featureType, expectedCount, detectedCount, status, severity)"
                + " VALUES (?, ?, 4, 3, 'MISSING', 2)", new Object[]{inspectionId, type});
    }

    private void insertSlotResult(long inspectionId, int slotId) {
        db.execSQL("INSERT INTO SlotResult"
                + " (inspectionId, slotId, expectedType, foundType, status, x, y, relX, relY, zone)"
                + " VALUES (?, ?, 'Bolt', 'Bolt', 'PRESENT', 0.5, 0.5, 0.25, 0.75, 'upper right')",
                new Object[]{inspectionId, slotId});
    }

    private void insertEvent(long inspectionId, String partId, boolean reviewed) {
        db.execSQL("INSERT INTO ClassificationEvent"
                + " (partId, inspectionId, timestamp, cropPath, predictedType, predictedScore,"
                + "  runnerUpType, runnerUpScore, reason, similarityThreshold, marginThreshold,"
                + "  encoderId, confirmedType, reviewed, reviewSource)"
                + " VALUES (?, ?, 1700000000000, '/crops/a.png', 'Bolt', 0.81, 'Screw', 0.62,"
                + "  'ACCEPTED', 0.7, 0.05, 'dinov2', 'Bolt', ?, 'spot_check')",
                new Object[]{partId, inspectionId, reviewed ? 1 : 0});
    }

    private long count(String sql) {
        try (Cursor c = db.query(sql)) {
            assertTrue(c.moveToNext());
            return c.getLong(0);
        }
    }

    // ── ClassificationEvent: the rows that must survive ─────────────────────

    /**
     * The one that matters most. These are human-labelled crops — the set ThresholdTuner fits
     * against — and the whole reason this table gets SET NULL rather than CASCADE. If someone
     * later "simplifies" the migration to treat all three tables alike, this fails.
     */
    @Test
    public void unassignedEventsSurviveWithTheirSentinelTurnedIntoNull() {
        insertEvent(-1, "CHASSIS-8821-B", true);
        insertEvent(-1, "CHASSIS-8821-B", false);

        AppDatabase.MIGRATION_10_11.migrate(db);

        assertEquals(2, count("SELECT COUNT(*) FROM ClassificationEvent"));
        assertEquals(2, count("SELECT COUNT(*) FROM ClassificationEvent"
                + " WHERE inspectionId IS NULL"));
    }

    /** An event whose inspection was deleted at some point is the same case, not a lost row. */
    @Test
    public void eventsPointingAtAVanishedInspectionAreDetachedRatherThanDropped() {
        insertEvent(999, "CHASSIS-8821-B", true);

        AppDatabase.MIGRATION_10_11.migrate(db);

        assertEquals(1, count("SELECT COUNT(*) FROM ClassificationEvent"));
        assertEquals(1, count("SELECT COUNT(*) FROM ClassificationEvent"
                + " WHERE inspectionId IS NULL"));
    }

    @Test
    public void eventsWithALiveInspectionKeepIt() {
        insertInspection(1);
        insertEvent(1, "CHASSIS-8821-B", true);

        AppDatabase.MIGRATION_10_11.migrate(db);

        assertEquals(1, count("SELECT COUNT(*) FROM ClassificationEvent WHERE inspectionId = 1"));
    }

    @Test
    public void everyEventColumnSurvivesTheRebuild() {
        insertInspection(1);
        insertEvent(1, "CHASSIS-8821-B", true);

        AppDatabase.MIGRATION_10_11.migrate(db);

        try (Cursor c = db.query("SELECT partId, timestamp, cropPath, predictedType,"
                + " predictedScore, runnerUpType, runnerUpScore, reason, similarityThreshold,"
                + " marginThreshold, encoderId, confirmedType, reviewed, reviewSource"
                + " FROM ClassificationEvent")) {
            assertTrue(c.moveToNext());
            assertEquals("CHASSIS-8821-B", c.getString(0));
            assertEquals(1_700_000_000_000L, c.getLong(1));
            assertEquals("/crops/a.png", c.getString(2));
            assertEquals("Bolt", c.getString(3));
            assertEquals(0.81f, c.getFloat(4), 1e-6);
            assertEquals("Screw", c.getString(5));
            assertEquals(0.62f, c.getFloat(6), 1e-6);
            assertEquals("ACCEPTED", c.getString(7));
            assertEquals(0.7f, c.getFloat(8), 1e-6);
            assertEquals(0.05f, c.getFloat(9), 1e-6);
            assertEquals("dinov2", c.getString(10));
            assertEquals("Bolt", c.getString(11));
            assertEquals(1, c.getInt(12));
            assertEquals("spot_check", c.getString(13));
        }
    }

    // ── Report tables: the rows that must not survive ───────────────────────

    @Test
    public void orphanReportRowsAreDroppedAndLiveOnesKept() {
        insertInspection(1);
        insertFeatureResult(1, "Bolt");
        insertFeatureResult(404, "Rivet");
        insertSlotResult(1, 1);
        insertSlotResult(404, 2);

        AppDatabase.MIGRATION_10_11.migrate(db);

        assertEquals(1, count("SELECT COUNT(*) FROM FeatureResult"));
        assertEquals(1, count("SELECT COUNT(*) FROM FeatureResult WHERE featureType = 'Bolt'"));
        assertEquals(1, count("SELECT COUNT(*) FROM SlotResult"));
        assertEquals(1, count("SELECT COUNT(*) FROM SlotResult WHERE slotId = 1"));
    }

    @Test
    public void everyReportColumnSurvivesTheRebuild() {
        insertInspection(1);
        insertFeatureResult(1, "Bolt");
        insertSlotResult(1, 3);

        AppDatabase.MIGRATION_10_11.migrate(db);

        try (Cursor c = db.query("SELECT featureType, expectedCount, detectedCount, status,"
                + " severity FROM FeatureResult")) {
            assertTrue(c.moveToNext());
            assertEquals("Bolt", c.getString(0));
            assertEquals(4, c.getInt(1));
            assertEquals(3, c.getInt(2));
            assertEquals("MISSING", c.getString(3));
            assertEquals(2, c.getInt(4));
        }
        try (Cursor c = db.query("SELECT slotId, expectedType, foundType, status, x, y,"
                + " relX, relY, zone, offsetPct FROM SlotResult")) {
            assertTrue(c.moveToNext());
            assertEquals(3, c.getInt(0));
            assertEquals("Bolt", c.getString(1));
            assertEquals("Bolt", c.getString(2));
            assertEquals("PRESENT", c.getString(3));
            assertEquals(0.25f, c.getFloat(6), 1e-6);
            assertEquals(0.75f, c.getFloat(7), 1e-6);
            assertEquals("upper right", c.getString(8));
            assertNull(c.getString(9));
        }
    }

    // ── The constraint has to be satisfiable ────────────────────────────────

    /**
     * Room runs this same check after migrating. A row it objects to is not a warning — the
     * database refuses to open, so the app is dead on launch after an upgrade.
     */
    @Test
    public void foreignKeyCheckIsCleanEvenOnAMessyDatabase() {
        insertInspection(1);
        insertFeatureResult(1, "Bolt");
        insertFeatureResult(404, "Rivet");
        insertSlotResult(404, 2);
        insertEvent(-1, "CHASSIS-8821-B", true);
        insertEvent(999, "CHASSIS-8821-B", false);
        insertEvent(1, "CHASSIS-8821-B", true);

        AppDatabase.MIGRATION_10_11.migrate(db);

        try (Cursor c = db.query("PRAGMA foreign_key_check")) {
            assertEquals("foreign_key_check reported violations", 0, c.getCount());
        }
    }

    @Test
    public void indicesExistAfterMigrating() {
        AppDatabase.MIGRATION_10_11.migrate(db);

        assertTrue(hasIndex("FeatureResult", "index_FeatureResult_inspectionId"));
        assertTrue(hasIndex("SlotResult", "index_SlotResult_inspectionId"));
        assertTrue(hasIndex("ClassificationEvent", "index_ClassificationEvent_inspectionId"));
        assertTrue(hasIndex("ClassificationEvent",
                "index_ClassificationEvent_partId_reviewed_timestamp"));
        assertTrue(hasIndex("InspectionRecord", "index_InspectionRecord_partId"));
    }

    @Test
    public void migratingAnEmptyDatabaseIsFine() {
        AppDatabase.MIGRATION_10_11.migrate(db);

        assertEquals(0, count("SELECT COUNT(*) FROM FeatureResult"));
        assertEquals(0, count("SELECT COUNT(*) FROM SlotResult"));
        assertEquals(0, count("SELECT COUNT(*) FROM ClassificationEvent"));
    }

    private boolean hasIndex(String table, String indexName) {
        try (Cursor c = db.query("PRAGMA index_list(" + table + ")")) {
            while (c.moveToNext()) {
                if (indexName.equals(c.getString(c.getColumnIndex("name")))) return true;
            }
        }
        return false;
    }
}
