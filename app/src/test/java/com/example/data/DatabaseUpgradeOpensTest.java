package com.example.data;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.database.Cursor;

import androidx.room.Room;
import androidx.sqlite.db.SupportSQLiteDatabase;
import androidx.sqlite.db.SupportSQLiteOpenHelper;
import androidx.sqlite.db.framework.FrameworkSQLiteOpenHelperFactory;
import androidx.test.core.app.ApplicationProvider;

import com.example.data.entity.ClassificationEvent;
import com.example.data.entity.SlotResult;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.util.ArrayList;
import java.util.List;

/**
 * Opens a database seeded with older data through Room itself, migrations and all.
 *
 * <p>{@link ChildTableMigrationTest} calls the migration directly, which proves the SQL does
 * what it should but skips everything Room does around it — the version bookkeeping, the schema
 * identity check, and the {@code PRAGMA foreign_key_check} that runs afterwards. That last one
 * is why this test exists: a migration that leaves one orphan row does not corrupt data or log a
 * warning, it stops the database opening, and the operator sees the app die on launch with a day
 * of collected inspections inside it.
 *
 * <p>The migration list comes from {@link AppDatabase#MIGRATIONS} rather than being repeated
 * here, so adding a migration and forgetting this test cannot leave it silently passing.
 */
@RunWith(RobolectricTestRunner.class)
public class DatabaseUpgradeOpensTest {

    private static final String DB_NAME = "upgrade_test_db";

    private Context context;
    private AppDatabase opened;

    @Before
    public void setUp() {
        context = ApplicationProvider.getApplicationContext();
        context.deleteDatabase(DB_NAME);
    }

    @After
    public void tearDown() {
        if (opened != null) opened.close();
        context.deleteDatabase(DB_NAME);
    }

    /** Writes a v10 database to disk, complete with the mess a real device accumulates. */
    private void seedV10() {
        SupportSQLiteOpenHelper helper = new FrameworkSQLiteOpenHelperFactory().create(
                SupportSQLiteOpenHelper.Configuration.builder(context)
                        .name(DB_NAME)
                        .callback(new SupportSQLiteOpenHelper.Callback(10) {
                            @Override
                            public void onCreate(SupportSQLiteDatabase db) {
                                createV10Schema(db);
                                seedRows(db);
                            }

                            @Override
                            public void onUpgrade(SupportSQLiteDatabase db, int from, int to) {
                            }
                        })
                        .build());
        helper.getWritableDatabase();
        helper.close();
    }

    private static void createV10Schema(SupportSQLiteDatabase db) {
        db.execSQL("CREATE TABLE IF NOT EXISTS InspectionRecord ("
                + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                + "partId TEXT, timestamp INTEGER NOT NULL, passOrFail TEXT, imageUri TEXT,"
                + "notes TEXT, uuid TEXT, digest TEXT, schemaVersion INTEGER,"
                + "supersedesUuid TEXT, operator TEXT, canonicalPartId TEXT)");
        db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS index_InspectionRecord_uuid"
                + " ON InspectionRecord (uuid)");
        db.execSQL("CREATE INDEX IF NOT EXISTS index_InspectionRecord_canonicalPartId"
                + " ON InspectionRecord (canonicalPartId)");
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
        db.execSQL("CREATE TABLE IF NOT EXISTS ReferenceProfile ("
                + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                + "partType TEXT, imageUri TEXT, createdAt INTEGER NOT NULL)");
        db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS index_ReferenceProfile_partType"
                + " ON ReferenceProfile (partType)");
    }

    /**
     * One good inspection plus every kind of dangling row a v10 device can be holding: report
     * fragments whose inspection is gone, an event that was never attached, and an event
     * pointing at an inspection that no longer exists.
     */
    private static void seedRows(SupportSQLiteDatabase db) {
        db.execSQL("INSERT INTO InspectionRecord (id, partId, timestamp, passOrFail, uuid,"
                + " canonicalPartId) VALUES (1, 'CHASSIS-8821-B', 1700000000000, 'FAIL',"
                + " '3f2504e0-4f89-41d3-9a0c-0305e82c3301', 'CHASSIS-8821-B')");

        db.execSQL("INSERT INTO FeatureResult"
                + " (inspectionId, featureType, expectedCount, detectedCount, status, severity)"
                + " VALUES (1, 'Bolt', 4, 3, 'MISSING', 2)");
        db.execSQL("INSERT INTO FeatureResult"
                + " (inspectionId, featureType, expectedCount, detectedCount, status, severity)"
                + " VALUES (777, 'Rivet', 2, 2, 'MATCH', 2)");

        db.execSQL("INSERT INTO SlotResult"
                + " (inspectionId, slotId, expectedType, foundType, status, x, y, relX, relY, zone)"
                + " VALUES (1, 1, 'Bolt', NULL, 'MISSING', 0.5, 0.5, 0.25, 0.75, 'upper right')");
        db.execSQL("INSERT INTO SlotResult"
                + " (inspectionId, slotId, expectedType, foundType, status, x, y, relX, relY, zone)"
                + " VALUES (777, 9, 'Bolt', 'Bolt', 'PRESENT', 0.1, 0.1, 0.1, 0.1, 'upper left')");

        for (Object[] row : new Object[][]{{1L, 1}, {-1L, 1}, {888L, 1}, {-1L, 0}}) {
            db.execSQL("INSERT INTO ClassificationEvent"
                    + " (partId, inspectionId, timestamp, cropPath, predictedType, predictedScore,"
                    + "  runnerUpType, runnerUpScore, reason, similarityThreshold,"
                    + "  marginThreshold, encoderId, confirmedType, reviewed, reviewSource)"
                    + " VALUES ('CHASSIS-8821-B', ?, 1700000000000, '/crops/a.png', 'Bolt', 0.81,"
                    + "  'Screw', 0.62, 'ACCEPTED', 0.7, 0.05, 'dinov2', 'Bolt', ?, 'spot_check')",
                    row);
        }
    }

    private AppDatabase openThroughRoom() {
        opened = Room.databaseBuilder(context, AppDatabase.class, DB_NAME)
                .addMigrations(AppDatabase.MIGRATIONS)
                .allowMainThreadQueries()
                .build();
        // Room is lazy: nothing migrates or validates until the database is actually touched.
        opened.inspectionDao().getAllInspections();
        return opened;
    }

    /**
     * The whole test in one line. If the migration leaves a single orphan, Room's
     * {@code foreign_key_check} throws here and the app would have died on launch.
     */
    @Test
    public void aV10DatabaseFullOfDanglingRowsStillOpens() {
        seedV10();
        AppDatabase db = openThroughRoom();

        assertEquals(1, db.inspectionDao().getAllInspections().size());
    }

    /**
     * v11→12 seeds the presets into the registry and v13→14 gives two of them their declared
     * bill of features. A device that upgrades through both must come out able to inspect
     * either preset against four features without any setup — and Room's schema validation on
     * open is what proves the migration's CREATE TABLE matches the entity.
     */
    @Test
    public void thePresetsArriveWithTheirDeclaredCounts() {
        seedV10();
        AppDatabase db = openThroughRoom();

        assertEquals(4, totalDeclared(db, "CHASSIS-8821-B"));
        assertEquals(4, totalDeclared(db, "PART-CHASSIS-A03"));
        assertEquals("no declaration was ever specified for this preset",
                0, totalDeclared(db, "ENGINE-COV-X42"));
    }

    private static int totalDeclared(AppDatabase db, String partId) {
        int total = 0;
        for (com.example.data.entity.ExpectedFeature f
                : db.expectedFeatureDao().getForPart(partId)) {
            total += f.getCount();
        }
        return total;
    }

    @Test
    public void theSurvivingInspectionKeepsItsReportIntact() {
        seedV10();
        AppDatabase db = openThroughRoom();

        assertEquals(1, db.inspectionDao().getFeatureResultsForInspection(1).size());
        assertEquals("Bolt",
                db.inspectionDao().getFeatureResultsForInspection(1).get(0).getFeatureType());
        assertEquals(1, db.inspectionDao().getSlotResultsForInspection(1).size());

        assertNotNull(db.inspectionDao().getInspectionByUuid(
                "3f2504e0-4f89-41d3-9a0c-0305e82c3301"));
    }

    @Test
    public void danglingReportFragmentsAreGone() {
        seedV10();
        AppDatabase db = openThroughRoom();

        assertEquals(0, db.inspectionDao().getFeatureResultsForInspection(777).size());
        assertEquals(0, db.inspectionDao().getSlotResultsForInspection(777).size());
    }

    /**
     * Four events went in; four come out. Three of them had no valid inspection, and losing any
     * of those would be losing labelled training data to a schema change.
     */
    @Test
    public void everyClassificationEventSurvivesTheUpgrade() {
        seedV10();
        AppDatabase db = openThroughRoom();

        assertEquals(4, db.classificationEventDao().getAllForPart("CHASSIS-8821-B").size());
        assertEquals(3, db.classificationEventDao().countReviewedForPart("CHASSIS-8821-B"));

        List<ClassificationEvent> reviewed =
                db.classificationEventDao().getRecentReviewedForPart("CHASSIS-8821-B", 50);
        assertEquals(3, reviewed.size());

        int attached = 0;
        for (ClassificationEvent e : db.classificationEventDao().getAllForPart("CHASSIS-8821-B")) {
            if (e.getInspectionId() != null) attached++;
        }
        assertEquals("only the event with a live inspection should still be attached",
                1, attached);
    }

    @Test
    public void cascadeWorksOnTheUpgradedDatabaseToo() {
        seedV10();
        AppDatabase db = openThroughRoom();

        db.inspectionDao().deleteInspection(1);

        assertEquals(0, db.inspectionDao().getFeatureResultsForInspection(1).size());
        assertEquals(0, db.inspectionDao().getSlotResultsForInspection(1).size());
        // And the training data is still there, now fully detached.
        assertEquals(4, db.classificationEventDao().getAllForPart("CHASSIS-8821-B").size());
        for (ClassificationEvent e : db.classificationEventDao().getAllForPart("CHASSIS-8821-B")) {
            assertNull(e.getInspectionId());
        }
    }

    @Test
    public void theUpgradedDatabaseReportsNoForeignKeyViolations() {
        seedV10();
        AppDatabase db = openThroughRoom();

        try (Cursor c = db.getOpenHelper().getReadableDatabase()
                .query("PRAGMA foreign_key_check")) {
            assertEquals(0, c.getCount());
        }
        try (Cursor c = db.getOpenHelper().getReadableDatabase().query("PRAGMA foreign_keys")) {
            assertTrue(c.moveToNext());
            assertEquals("Room should have enforcement switched on", 1, c.getInt(0));
        }
    }

    /**
     * v13 added {@code sizeRatio} to SlotResult. The pre-existing row must read as unjudged —
     * null, not 0 or 1 — and a row filed after the upgrade must keep the value it was given.
     */
    @Test
    public void slotSizeRatioIsNullOnOldRowsAndKeptOnNewOnes() {
        seedV10();
        AppDatabase db = openThroughRoom();

        List<SlotResult> before = db.inspectionDao().getSlotResultsForInspection(1);
        assertEquals(1, before.size());
        assertNull("nothing was size-checked before the column existed", before.get(0).getSizeRatio());

        List<SlotResult> rows = new ArrayList<>();
        rows.add(new SlotResult(1, 2, "Bolt", "Bolt", "PRESENT", 0.5f, 0.5f, 0.5f, 0.5f,
                null, "centre", 1.42f));
        db.inspectionDao().insertSlotResults(rows);

        Float stored = null;
        for (SlotResult s : db.inspectionDao().getSlotResultsForInspection(1)) {
            if (s.getSlotId() == 2) stored = s.getSizeRatio();
        }
        assertNotNull(stored);
        assertEquals(1.42f, stored, 1e-6f);
    }
}
