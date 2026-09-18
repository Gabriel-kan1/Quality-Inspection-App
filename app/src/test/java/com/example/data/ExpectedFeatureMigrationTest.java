package com.example.data;

import static org.junit.Assert.assertEquals;
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

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Covers the v13 → v14 upgrade, which adds the operator-declared feature counts.
 *
 * <p>Two things are worth pinning. The presets that ship with a known bill of features must
 * come out of the upgrade carrying it — that is what lets an upgraded device inspect
 * CHASSIS-8821-B against four features without anyone re-doing setup. And a preset the
 * operator has already deleted must <em>not</em> come back as a set of counts for a part that
 * is no longer in the registry.
 */
@RunWith(RobolectricTestRunner.class)
public class ExpectedFeatureMigrationTest {

    /** PartProfile as v13 left it — the only table this migration reads. */
    private static final String V13_SCHEMA =
            "CREATE TABLE IF NOT EXISTS PartProfile ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                    + "productId INTEGER NOT NULL,"
                    + "partId TEXT,"
                    + "displayName TEXT,"
                    + "createdAt INTEGER NOT NULL)";

    private SupportSQLiteOpenHelper helper;
    private SupportSQLiteDatabase db;

    @Before
    public void setUp() {
        Context context = ApplicationProvider.getApplicationContext();
        helper = new FrameworkSQLiteOpenHelperFactory().create(
                SupportSQLiteOpenHelper.Configuration.builder(context)
                        .name(null)
                        .callback(new SupportSQLiteOpenHelper.Callback(13) {
                            @Override
                            public void onCreate(SupportSQLiteDatabase database) {
                                database.execSQL(V13_SCHEMA);
                            }

                            @Override
                            public void onUpgrade(SupportSQLiteDatabase database, int f, int t) {
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

    private void registerPart(String partId) {
        db.execSQL("INSERT INTO PartProfile (productId, partId, displayName, createdAt)"
                + " VALUES (1, ?, NULL, 100)", new Object[]{partId});
    }

    private Map<String, Integer> declaredFor(String partId) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        try (Cursor c = db.query("SELECT featureType, count FROM ExpectedFeature"
                + " WHERE partId = ? ORDER BY id", new Object[]{partId})) {
            while (c.moveToNext()) counts.put(c.getString(0), c.getInt(1));
        }
        return counts;
    }

    @Test
    public void theRegisteredPresetGetsItsFourFeatures() {
        registerPart("CHASSIS-8821-B");
        AppDatabase.MIGRATION_13_14.migrate(db);

        Map<String, Integer> counts = declaredFor("CHASSIS-8821-B");
        assertEquals(Integer.valueOf(1), counts.get("hole"));
        assertEquals(Integer.valueOf(2), counts.get("stud"));
        assertEquals(Integer.valueOf(1), counts.get("screw"));
        assertEquals("one hole, two studs, one screw", 3, counts.size());
    }

    @Test
    public void theOtherPresetGetsABoltHeadRatherThanAScrew() {
        registerPart("PART-CHASSIS-A03");
        AppDatabase.MIGRATION_13_14.migrate(db);

        Map<String, Integer> counts = declaredFor("PART-CHASSIS-A03");
        assertEquals(Integer.valueOf(1), counts.get("hole"));
        assertEquals(Integer.valueOf(2), counts.get("stud"));
        assertEquals(Integer.valueOf(1), counts.get("bolt_head"));
        assertEquals(3, counts.size());
    }

    @Test
    public void aPresetTheOperatorDeletedDoesNotComeBackAsCounts() {
        registerPart("CHASSIS-8821-B");
        // PART-CHASSIS-A03 is absent: the operator removed it before upgrading.
        AppDatabase.MIGRATION_13_14.migrate(db);

        assertTrue("counts for a part that no longer exists",
                declaredFor("PART-CHASSIS-A03").isEmpty());
    }

    @Test
    public void runningTwiceChangesNothing() {
        registerPart("CHASSIS-8821-B");
        AppDatabase.MIGRATION_13_14.migrate(db);
        AppDatabase.MIGRATION_13_14.migrate(db);

        try (Cursor c = db.query("SELECT COUNT(*) FROM ExpectedFeature")) {
            c.moveToFirst();
            assertEquals(3, c.getInt(0));
        }
    }

    @Test
    public void thePinnedSeedTotalsFourPerPart() {
        // The user's specification: four features on each of the two presets. If the constant
        // drifts, the migration and the fresh-install seed drift with it.
        Map<String, Integer> totals = new LinkedHashMap<>();
        for (Object[] row : AppDatabase.SEED_EXPECTED_FEATURES) {
            String partId = (String) row[0];
            Integer prev = totals.get(partId);
            totals.put(partId, (prev == null ? 0 : prev) + (Integer) row[2]);
        }
        assertEquals(Integer.valueOf(4), totals.get("CHASSIS-8821-B"));
        assertEquals(Integer.valueOf(4), totals.get("PART-CHASSIS-A03"));
        assertEquals("only the two presets carry a declaration", 2, totals.size());
    }
}
