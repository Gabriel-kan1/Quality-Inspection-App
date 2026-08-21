package com.example.data;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import android.content.Context;
import android.database.Cursor;

import androidx.sqlite.db.SupportSQLiteDatabase;
import androidx.sqlite.db.SupportSQLiteOpenHelper;
import androidx.sqlite.db.framework.FrameworkSQLiteOpenHelperFactory;
import androidx.test.core.app.ApplicationProvider;

import com.example.ml.PartFiles;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

/**
 * Covers the v9 → v10 upgrade, which is the one that has to run on a device holding real
 * inspection history.
 *
 * <p>Driven against a raw SQLite database rather than through Room's {@code MigrationTestHelper}
 * because that helper reads exported schema JSON, and this project builds with
 * {@code exportSchema = false}. What is worth testing here is the backfill, not Room's own
 * schema diffing: the migration writes two values per existing row that no SQL expression could
 * produce, and a mistake there is silent — the app opens fine and the data is quietly wrong.
 */
@RunWith(RobolectricTestRunner.class)
public class InspectionRecordMigrationTest {

    /** The InspectionRecord table exactly as v9 left it. */
    private static final String V9_SCHEMA =
            "CREATE TABLE IF NOT EXISTS InspectionRecord ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                    + "partId TEXT,"
                    + "timestamp INTEGER NOT NULL,"
                    + "passOrFail TEXT,"
                    + "imageUri TEXT,"
                    + "notes TEXT)";

    private SupportSQLiteOpenHelper helper;
    private SupportSQLiteDatabase db;

    @Before
    public void setUp() {
        Context context = ApplicationProvider.getApplicationContext();
        helper = new FrameworkSQLiteOpenHelperFactory().create(
                SupportSQLiteOpenHelper.Configuration.builder(context)
                        // Null name is an in-memory database, so each test starts clean.
                        .name(null)
                        .callback(new SupportSQLiteOpenHelper.Callback(9) {
                            @Override
                            public void onCreate(SupportSQLiteDatabase database) {
                                database.execSQL(V9_SCHEMA);
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

    private void insertV9Row(String partId, String verdict) {
        db.execSQL("INSERT INTO InspectionRecord (partId, timestamp, passOrFail, imageUri, notes)"
                        + " VALUES (?, ?, ?, ?, ?)",
                new Object[]{partId, 1_700_000_000_000L, verdict, "file:///x.jpg", "note"});
    }

    @Test
    public void existingRowsSurviveAndKeepTheirValues() {
        insertV9Row("CHASSIS-8821-B", "PASS");
        insertV9Row("ENGINE-COV-X42", "FAIL");

        AppDatabase.MIGRATION_9_10.migrate(db);

        try (Cursor c = db.query("SELECT partId, passOrFail, notes FROM InspectionRecord"
                + " ORDER BY id")) {
            assertEquals(2, c.getCount());
            assertTrue(c.moveToNext());
            assertEquals("CHASSIS-8821-B", c.getString(0));
            assertEquals("PASS", c.getString(1));
            assertEquals("note", c.getString(2));
            assertTrue(c.moveToNext());
            assertEquals("ENGINE-COV-X42", c.getString(0));
            assertEquals("FAIL", c.getString(1));
        }
    }

    @Test
    public void everyExistingRowGetsADistinctUsableUuid() {
        for (int i = 0; i < 25; i++) {
            insertV9Row("PART-" + i, "PASS");
        }

        AppDatabase.MIGRATION_9_10.migrate(db);

        Set<String> seen = new HashSet<>();
        try (Cursor c = db.query("SELECT uuid FROM InspectionRecord")) {
            while (c.moveToNext()) {
                String uuid = c.getString(0);
                assertNotNull("row left without a uuid", uuid);
                // Parseable, because ReportCode packs it as sixteen raw bytes.
                assertEquals(uuid, UUID.fromString(uuid).toString());
                assertTrue("duplicate uuid " + uuid, seen.add(uuid));
            }
        }
        assertEquals(25, seen.size());
    }

    /**
     * The backfill is the reason this runs in Java: {@code PartFiles.safeName} replaces every
     * character outside a whitelist, which stock SQLite has no way to express.
     */
    @Test
    public void canonicalPartIdIsBackfilledThroughTheSameSanitiserThePathsUse() {
        insertV9Row("A/B", "PASS");
        insertV9Row("A_B", "PASS");
        insertV9Row("chassis rev 2 (new)", "FAIL");
        insertV9Row(null, "UNVERIFIED");

        AppDatabase.MIGRATION_9_10.migrate(db);

        try (Cursor c = db.query("SELECT partId, canonicalPartId FROM InspectionRecord ORDER BY id")) {
            while (c.moveToNext()) {
                assertEquals(PartFiles.safeName(c.isNull(0) ? null : c.getString(0)),
                        c.getString(1));
            }
        }

        // The point of storing it: two raw ids that share every on-disk file for the part now
        // group together, which is unanswerable from partId alone.
        try (Cursor c = db.query("SELECT COUNT(*) FROM InspectionRecord"
                + " WHERE canonicalPartId = 'A_B'")) {
            assertTrue(c.moveToNext());
            assertEquals(2, c.getInt(0));
        }
    }

    /**
     * A digest claims "nothing has changed since this was filed". Computing one during the
     * migration would compute it over whatever the row holds now and assert a guarantee that
     * was never in force, so these stay null and the report says the record is unsealed.
     */
    @Test
    public void legacyRowsAreLeftUnsealedRatherThanBackdated() {
        insertV9Row("CHASSIS-8821-B", "PASS");

        AppDatabase.MIGRATION_9_10.migrate(db);

        try (Cursor c = db.query("SELECT digest, schemaVersion, supersedesUuid, operator"
                + " FROM InspectionRecord")) {
            assertTrue(c.moveToNext());
            assertNull(c.getString(0));
            assertTrue(c.isNull(1));
            assertNull(c.getString(2));
            assertNull(c.getString(3));
        }
    }

    @Test
    public void uuidIndexIsUniqueAfterMigrating() {
        insertV9Row("CHASSIS-8821-B", "PASS");
        AppDatabase.MIGRATION_9_10.migrate(db);

        String uuid;
        try (Cursor c = db.query("SELECT uuid FROM InspectionRecord")) {
            assertTrue(c.moveToNext());
            uuid = c.getString(0);
        }

        try {
            db.execSQL("INSERT INTO InspectionRecord (partId, timestamp, uuid)"
                    + " VALUES ('X', 1, ?)", new Object[]{uuid});
            fail("a second row took the same uuid");
        } catch (RuntimeException expected) {
            // The unique index is what makes the uuid safe to look a report up by.
        }
    }

    @Test
    public void migratingAnEmptyDatabaseIsFine() {
        AppDatabase.MIGRATION_9_10.migrate(db);

        try (Cursor c = db.query("SELECT COUNT(*) FROM InspectionRecord")) {
            assertTrue(c.moveToNext());
            assertEquals(0, c.getInt(0));
        }
        // Columns still have to exist, or the first insert after an upgrade on a fresh install
        // that never filed anything would fail.
        try (Cursor c = db.query("SELECT uuid, digest, schemaVersion, supersedesUuid, operator,"
                + " canonicalPartId FROM InspectionRecord")) {
            assertEquals(6, c.getColumnCount());
        }
    }
}
