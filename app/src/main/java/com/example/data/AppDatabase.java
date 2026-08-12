package com.example.data;

import android.content.Context;
import androidx.annotation.NonNull;
import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;
import androidx.room.migration.Migration;
import androidx.sqlite.db.SupportSQLiteDatabase;

import com.example.data.dao.ClassificationEventDao;
import com.example.data.dao.InspectionDao;
import com.example.data.dao.ReferenceProfileDao;
import com.example.data.entity.ClassificationEvent;
import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.ReferenceProfile;
import com.example.data.entity.SlotResult;

@Database(entities = {InspectionRecord.class, FeatureResult.class, ReferenceProfile.class,
        ClassificationEvent.class, SlotResult.class}, version = 9, exportSchema = false)
public abstract class AppDatabase extends RoomDatabase {
    private static volatile AppDatabase INSTANCE;

    public abstract InspectionDao inspectionDao();
    public abstract ReferenceProfileDao referenceProfileDao();
    public abstract ClassificationEventDao classificationEventDao();

    /**
     * Adds ClassificationEvent.encoderId so ThresholdTuner can tell which embedding space
     * a logged score came from.
     *
     * Written as a real migration rather than leaning on the destructive fallback: that
     * would drop every table, discarding the operator's entire inspection history to add
     * one column. Existing rows predate the DINOv2 swap, so they are stamped as legacy and
     * excluded from threshold fitting while remaining available as labelled data for a
     * future off-device fine-tune.
     *
     * The column is added without a SQL DEFAULT on purpose. Room validates the migrated
     * schema against the entity, and the field is a plain nullable String with no
     * {@code @ColumnInfo(defaultValue = ...)}; declaring a default here would not match and
     * would throw on open. The backfill is a separate UPDATE instead.
     */
    static final Migration MIGRATION_3_4 = new Migration(3, 4) {
        @Override
        public void migrate(@NonNull SupportSQLiteDatabase db) {
            db.execSQL("ALTER TABLE ClassificationEvent ADD COLUMN encoderId TEXT");
            db.execSQL("UPDATE ClassificationEvent SET encoderId = '"
                    + ClassificationEvent.ENCODER_LEGACY_UNTAGGED + "'");
        }
    };

    /**
     * Makes ReferenceProfile.partType unique, so saving a reference photo replaces the part's
     * previous one instead of stacking another row beside it.
     *
     * Duplicates already exist in the wild and have to go before the index can be created.
     * The survivor per part is chosen deliberately rather than by id: a real photo always
     * beats a {@code dummy_placeholder}, and only then does the newest win. Picking purely by
     * newest would keep a placeholder written by a later "skip setup" and preserve exactly the
     * blocked state this migration exists to clear.
     *
     * Window functions would express this more directly but are unavailable on the SQLite
     * shipped with API 24, so it is a correlated subquery.
     */
    static final Migration MIGRATION_4_5 = new Migration(4, 5) {
        @Override
        public void migrate(@NonNull SupportSQLiteDatabase db) {
            db.execSQL("DELETE FROM ReferenceProfile WHERE id NOT IN ("
                    + "  SELECT ("
                    + "    SELECT p2.id FROM ReferenceProfile p2"
                    + "    WHERE p2.partType = p1.partType"
                    + "    ORDER BY CASE WHEN p2.imageUri = 'dummy_placeholder' THEN 1 ELSE 0 END,"
                    + "             p2.createdAt DESC, p2.id DESC"
                    + "    LIMIT 1"
                    + "  ) FROM ReferenceProfile p1 GROUP BY p1.partType"
                    + ")");
            db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS index_ReferenceProfile_partType"
                    + " ON ReferenceProfile (partType)");
        }
    };

    /**
     * Replaces FeatureResult.critical with FeatureResult.severity, the three-stop scale
     * (0 ignored / 1 advisory / 2 critical) the report now branches on.
     *
     * SQLite before 3.35 cannot drop a column, so this is the standard rebuild-and-rename
     * rather than an ALTER. Existing rows carry the same information a boolean could:
     * critical=1 maps to CRITICAL, critical=0 to ADVISORY. Nothing maps to IGNORED —
     * demoting a historical row to "not reported at all" would rewrite what a filed
     * inspection said, and the old value cannot distinguish the two cases anyway.
     *
     * As in {@link #MIGRATION_3_4}, the new column is declared without a SQL DEFAULT: the
     * entity field has no {@code @ColumnInfo(defaultValue = ...)}, and Room fails the
     * schema validation on open if the two disagree. The INSERT supplies every value.
     */
    static final Migration MIGRATION_5_6 = new Migration(5, 6) {
        @Override
        public void migrate(@NonNull SupportSQLiteDatabase db) {
            db.execSQL("CREATE TABLE IF NOT EXISTS FeatureResult_new ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                    + "inspectionId INTEGER NOT NULL,"
                    + "featureType TEXT,"
                    + "expectedCount INTEGER NOT NULL,"
                    + "detectedCount INTEGER NOT NULL,"
                    + "status TEXT,"
                    + "severity INTEGER NOT NULL)");
            db.execSQL("INSERT INTO FeatureResult_new"
                    + " (id, inspectionId, featureType, expectedCount, detectedCount, status, severity)"
                    + " SELECT id, inspectionId, featureType, expectedCount, detectedCount, status,"
                    + "        CASE WHEN critical = 0 THEN 1 ELSE 2 END"
                    + " FROM FeatureResult");
            db.execSQL("DROP TABLE FeatureResult");
            db.execSQL("ALTER TABLE FeatureResult_new RENAME TO FeatureResult");
        }
    };

    /**
     * Adds the SlotResult table, which records per-position localisation findings for
     * critical features — which specific expected fastener is missing and where, rather than
     * only how many of each type were counted.
     *
     * Purely additive, so no rebuild and nothing to backfill: inspections filed before
     * positional checking existed simply have no slot rows, and the report renders that as
     * "not available for this inspection" rather than as an all-clear.
     */
    static final Migration MIGRATION_6_7 = new Migration(6, 7) {
        @Override
        public void migrate(@NonNull SupportSQLiteDatabase db) {
            db.execSQL("CREATE TABLE IF NOT EXISTS SlotResult ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                    + "inspectionId INTEGER NOT NULL,"
                    + "slotId INTEGER NOT NULL,"
                    + "expectedType TEXT,"
                    + "foundType TEXT,"
                    + "status TEXT,"
                    + "x REAL NOT NULL,"
                    + "y REAL NOT NULL,"
                    + "zone TEXT)");
        }
    };

    /**
     * Adds SlotResult.relX/relY — positions normalised to the part's own extent rather than to
     * the reference photo's frame, which is what the localisation report now shows.
     *
     * A table rebuild rather than ALTER TABLE: SQLite cannot add a NOT NULL column without a
     * SQL default, and a default is exactly what Room's schema validation forbids here (see
     * {@link #MIGRATION_3_4}).
     *
     * Existing rows are backfilled from x/y. That is not a claim they were normalised — those
     * inspections predate the envelope entirely, and the frame coordinate is precisely what
     * their report displayed. Copying it preserves what was said rather than back-computing a
     * normalisation that was never applied.
     */
    static final Migration MIGRATION_7_8 = new Migration(7, 8) {
        @Override
        public void migrate(@NonNull SupportSQLiteDatabase db) {
            db.execSQL("CREATE TABLE IF NOT EXISTS SlotResult_new ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                    + "inspectionId INTEGER NOT NULL,"
                    + "slotId INTEGER NOT NULL,"
                    + "expectedType TEXT,"
                    + "foundType TEXT,"
                    + "status TEXT,"
                    + "x REAL NOT NULL,"
                    + "y REAL NOT NULL,"
                    + "relX REAL NOT NULL,"
                    + "relY REAL NOT NULL,"
                    + "zone TEXT)");
            db.execSQL("INSERT INTO SlotResult_new"
                    + " (id, inspectionId, slotId, expectedType, foundType, status, x, y, relX, relY, zone)"
                    + " SELECT id, inspectionId, slotId, expectedType, foundType, status, x, y, x, y, zone"
                    + " FROM SlotResult");
            db.execSQL("DROP TABLE SlotResult");
            db.execSQL("ALTER TABLE SlotResult_new RENAME TO SlotResult");
        }
    };

    /**
     * Adds SlotResult.offsetPct — how far a displaced feature sits from the position it
     * belongs to, so the report can say "it is here, 12% off" instead of "missing".
     *
     * A plain ALTER this time rather than the rebuild the other column additions needed: the
     * field is a boxed Float, so the column is nullable and SQLite does not demand a default
     * for existing rows. Those rows predate the distinction and correctly read as null —
     * nothing was ever measured for them.
     */
    static final Migration MIGRATION_8_9 = new Migration(8, 9) {
        @Override
        public void migrate(@NonNull SupportSQLiteDatabase db) {
            db.execSQL("ALTER TABLE SlotResult ADD COLUMN offsetPct REAL");
        }
    };

    public static AppDatabase getInstance(final Context context) {
        if (INSTANCE == null) {
            synchronized (AppDatabase.class) {
                if (INSTANCE == null) {
                    INSTANCE = Room.databaseBuilder(context.getApplicationContext(),
                                    AppDatabase.class, "shop_quality_inspection_db")
                            .addMigrations(MIGRATION_3_4, MIGRATION_4_5, MIGRATION_5_6, MIGRATION_6_7,
                                    MIGRATION_7_8, MIGRATION_8_9)
                            // Retained as a backstop for the older unversioned upgrade paths
                            // that shipped before MIGRATION_3_4 existed. Any NEW schema change
                            // should add its own Migration rather than rely on this.
                            .fallbackToDestructiveMigration()
                            .build();
                }
            }
        }
        return INSTANCE;
    }
}
