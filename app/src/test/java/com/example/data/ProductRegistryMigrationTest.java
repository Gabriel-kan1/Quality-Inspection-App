package com.example.data;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
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

import java.util.ArrayList;
import java.util.List;

/**
 * Covers the v11 → v12 upgrade, which introduces products and the part registry.
 *
 * <p>The thing worth pinning down is the backfill. Before this migration the part list was
 * assembled at render time from ReferenceProfile rows plus three hardcoded ids; afterwards it
 * comes entirely from the registry. If the backfill misses a part, that part simply vanishes
 * from the operator's list while its reference photo, learned layout and inspection history all
 * remain on disk — a silent loss, and the kind that is only noticed later.
 */
@RunWith(RobolectricTestRunner.class)
public class ProductRegistryMigrationTest {

    /** ReferenceProfile as v11 left it — the only table this migration reads. */
    private static final String V11_SCHEMA =
            "CREATE TABLE IF NOT EXISTS ReferenceProfile ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                    + "partType TEXT,"
                    + "imageUri TEXT,"
                    + "createdAt INTEGER NOT NULL)";

    private SupportSQLiteOpenHelper helper;
    private SupportSQLiteDatabase db;

    @Before
    public void setUp() {
        Context context = ApplicationProvider.getApplicationContext();
        helper = new FrameworkSQLiteOpenHelperFactory().create(
                SupportSQLiteOpenHelper.Configuration.builder(context)
                        .name(null)
                        .callback(new SupportSQLiteOpenHelper.Callback(11) {
                            @Override
                            public void onCreate(SupportSQLiteDatabase database) {
                                database.execSQL(V11_SCHEMA);
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

    private void insertReference(String partType, long createdAt) {
        db.execSQL("INSERT INTO ReferenceProfile (partType, imageUri, createdAt) VALUES (?, ?, ?)",
                new Object[]{partType, "file:///" + partType + ".jpg", createdAt});
    }

    private List<String> registeredPartIds() {
        List<String> ids = new ArrayList<>();
        try (Cursor c = db.query("SELECT partId FROM PartProfile ORDER BY id")) {
            while (c.moveToNext()) ids.add(c.getString(0));
        }
        return ids;
    }

    private long defaultProductId() {
        try (Cursor c = db.query("SELECT id FROM Product WHERE name = 'Uncategorised'")) {
            assertTrue("default product missing", c.moveToFirst());
            return c.getLong(0);
        }
    }

    @Test
    public void createsOneDefaultProductHoldingEverything() {
        insertReference("WIDGET-1", 100L);
        AppDatabase.MIGRATION_11_12.migrate(db);

        try (Cursor c = db.query("SELECT COUNT(*) FROM Product")) {
            c.moveToFirst();
            assertEquals("upgrade should create exactly one product", 1, c.getInt(0));
        }
        long productId = defaultProductId();
        try (Cursor c = db.query("SELECT COUNT(*) FROM PartProfile WHERE productId != ?",
                new Object[]{productId})) {
            c.moveToFirst();
            assertEquals("every part should land in the default product", 0, c.getInt(0));
        }
    }

    @Test
    public void everyPartWithAReferencePhotoSurvivesIntoTheRegistry() {
        insertReference("WIDGET-1", 100L);
        insertReference("WIDGET-2", 200L);

        AppDatabase.MIGRATION_11_12.migrate(db);

        List<String> ids = registeredPartIds();
        assertTrue("WIDGET-1 lost on upgrade", ids.contains("WIDGET-1"));
        assertTrue("WIDGET-2 lost on upgrade", ids.contains("WIDGET-2"));
    }

    @Test
    public void seedsThePresetsSoAFreshDeviceHasTheShortcuts() {
        AppDatabase.MIGRATION_11_12.migrate(db);

        List<String> ids = registeredPartIds();
        assertTrue(ids.contains("CHASSIS-8821-B"));
        assertTrue(ids.contains("PART-CHASSIS-A03"));
        assertTrue(ids.contains("ENGINE-COV-X42"));
    }

    /**
     * A preset that already has a reference photo must not end up in the list twice — once from
     * ReferenceProfile and once from the seed.
     */
    @Test
    public void doesNotDuplicateAPresetThatAlreadyHasAReference() {
        insertReference("CHASSIS-8821-B", 100L);

        AppDatabase.MIGRATION_11_12.migrate(db);

        int occurrences = 0;
        for (String id : registeredPartIds()) if ("CHASSIS-8821-B".equals(id)) occurrences++;
        assertEquals("preset with a saved reference was registered twice", 1, occurrences);
    }

    /**
     * Backfilled rows carry no label. Writing the id into displayName as well would make a part
     * nobody has renamed indistinguishable from one deliberately renamed to match its id, and
     * the list could no longer fall back cleanly.
     */
    @Test
    public void leavesBackfilledRowsWithoutADisplayName() {
        insertReference("WIDGET-1", 100L);
        AppDatabase.MIGRATION_11_12.migrate(db);

        try (Cursor c = db.query("SELECT displayName FROM PartProfile WHERE partId = 'WIDGET-1'")) {
            assertTrue(c.moveToFirst());
            assertNull("backfill should not invent a label", c.getString(0));
        }
    }

    @Test
    public void ignoresBlankAndNullPartTypes() {
        insertReference(null, 100L);
        insertReference("", 110L);
        insertReference("WIDGET-1", 120L);

        AppDatabase.MIGRATION_11_12.migrate(db);

        for (String id : registeredPartIds()) {
            assertNotNull(id);
            assertTrue("blank part id reached the registry", !id.isEmpty());
        }
    }

    /** Deleting a product takes its registry rows with it, and leaves no dangling productId. */
    @Test
    public void deletingAProductCascadesToItsParts() {
        insertReference("WIDGET-1", 100L);
        AppDatabase.MIGRATION_11_12.migrate(db);
        db.execSQL("PRAGMA foreign_keys = ON");

        db.execSQL("DELETE FROM Product WHERE id = ?", new Object[]{defaultProductId()});

        try (Cursor c = db.query("SELECT COUNT(*) FROM PartProfile")) {
            c.moveToFirst();
            assertEquals(0, c.getInt(0));
        }
    }
}
