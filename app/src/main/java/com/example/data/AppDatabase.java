package com.example.data;

import android.content.Context;
import android.database.Cursor;
import androidx.annotation.NonNull;
import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;
import androidx.room.migration.Migration;
import androidx.sqlite.db.SupportSQLiteDatabase;

import com.example.data.dao.ClassificationEventDao;
import com.example.data.dao.InspectionDao;
import com.example.data.dao.PartProfileDao;
import com.example.data.dao.ProductDao;
import com.example.data.dao.ReferenceProfileDao;
import com.example.data.entity.ClassificationEvent;
import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.PartProfile;
import com.example.data.entity.Product;
import com.example.data.entity.ReferenceProfile;
import com.example.data.entity.SlotResult;
import com.example.ml.PartFiles;

import java.util.UUID;

@Database(entities = {InspectionRecord.class, FeatureResult.class, ReferenceProfile.class,
        ClassificationEvent.class, SlotResult.class, Product.class, PartProfile.class},
        version = 12, exportSchema = false)
public abstract class AppDatabase extends RoomDatabase {
    private static volatile AppDatabase INSTANCE;

    /**
     * Duplicated from {@code Product.DEFAULT_NAME} on purpose: a migration has to keep writing
     * the value it wrote on the day it shipped. If the constant is later renamed, every device
     * that already ran this migration would still hold the old name, and a migration reading
     * the new one would create a second, empty default product beside it.
     */
    private static final String DEFAULT_PRODUCT_NAME = "Uncategorised";

    /** Likewise pinned: the preset ids as they were when the registry was introduced. */
    private static final String[] SEED_PART_IDS =
            {"CHASSIS-8821-B", "PART-CHASSIS-A03", "ENGINE-COV-X42"};

    public abstract InspectionDao inspectionDao();
    public abstract ReferenceProfileDao referenceProfileDao();
    public abstract ClassificationEventDao classificationEventDao();
    public abstract ProductDao productDao();
    public abstract PartProfileDao partProfileDao();

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

    /**
     * Gives a filed inspection an identity and an integrity seal: uuid, digest, schemaVersion,
     * supersedesUuid, operator and canonicalPartId.
     *
     * All six are added as plain ALTERs rather than the table rebuild the earlier column
     * additions needed, because all six columns are nullable — the two that would otherwise
     * have been primitives ({@code schemaVersion}) are boxed on the entity for exactly that
     * reason, the same trade {@link #MIGRATION_8_9} made for {@code offsetPct}. As in
     * {@link #MIGRATION_3_4}, none of them carries a SQL DEFAULT: the entity fields have no
     * {@code @ColumnInfo(defaultValue = ...)} and Room fails schema validation on open if the
     * two disagree.
     *
     * The backfill runs in Java rather than SQL because both values it writes are beyond what
     * stock SQLite can express — a real UUID, and {@link PartFiles#safeName} whose character
     * class has no regex equivalent here. There is one row per filed inspection, not one per
     * detection, so the cursor pass is cheap.
     *
     * {@code digest} and {@code schemaVersion} are deliberately left null on existing rows.
     * A digest claims "nothing has changed since this was filed"; computing one now would
     * compute it over whatever the row holds today and assert a guarantee that was never in
     * force. Those rows read as unsealed, which is the truth about them.
     */
    static final Migration MIGRATION_9_10 = new Migration(9, 10) {
        @Override
        public void migrate(@NonNull SupportSQLiteDatabase db) {
            db.execSQL("ALTER TABLE InspectionRecord ADD COLUMN uuid TEXT");
            db.execSQL("ALTER TABLE InspectionRecord ADD COLUMN digest TEXT");
            db.execSQL("ALTER TABLE InspectionRecord ADD COLUMN schemaVersion INTEGER");
            db.execSQL("ALTER TABLE InspectionRecord ADD COLUMN supersedesUuid TEXT");
            db.execSQL("ALTER TABLE InspectionRecord ADD COLUMN operator TEXT");
            db.execSQL("ALTER TABLE InspectionRecord ADD COLUMN canonicalPartId TEXT");

            try (Cursor c = db.query("SELECT id, partId FROM InspectionRecord")) {
                while (c.moveToNext()) {
                    long id = c.getLong(0);
                    String partId = c.isNull(1) ? null : c.getString(1);
                    db.execSQL("UPDATE InspectionRecord SET uuid = ?, canonicalPartId = ?"
                                    + " WHERE id = ?",
                            new Object[]{UUID.randomUUID().toString(),
                                    PartFiles.safeName(partId), id});
                }
            }

            // Created after the backfill, not before: the column starts out entirely null and
            // SQLite treats every null as distinct, so the index would be satisfied either way
            // — but building it once over final values beats maintaining it across the updates.
            db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS index_InspectionRecord_uuid"
                    + " ON InspectionRecord (uuid)");
            db.execSQL("CREATE INDEX IF NOT EXISTS index_InspectionRecord_canonicalPartId"
                    + " ON InspectionRecord (canonicalPartId)");
        }
    };

    /**
     * Gives the three child tables the foreign keys and indices they never had.
     *
     * <p>Until now nothing connected a FeatureResult, SlotResult or ClassificationEvent to a
     * live inspection, and every lookup by {@code inspectionId} was a full table scan — on
     * ClassificationEvent, which grows by one row per detection, that is the table the learning
     * loop reads on every single inspection.
     *
     * <p>The delete rule is deliberately not uniform. FeatureResult and SlotResult CASCADE
     * because they are the report; ClassificationEvent is SET NULL because it is labelled
     * training data that merely happened to be attached to one. See the entity for the full
     * argument.
     *
     * <p>SQLite cannot add a constraint with ALTER TABLE, so each table is rebuilt the way
     * {@link #MIGRATION_5_6} and {@link #MIGRATION_7_8} already do it. What is different here is
     * that the constraint has to be <em>satisfiable</em> before it exists: Room runs
     * {@code PRAGMA foreign_key_check} after migrating, and a device carrying real history that
     * fails it will not open at all. So every INSERT ... SELECT below repairs as it copies.
     */
    static final Migration MIGRATION_10_11 = new Migration(10, 11) {
        @Override
        public void migrate(@NonNull SupportSQLiteDatabase db) {
            // ── FeatureResult: cascade, so orphans go ──────────────────────────
            db.execSQL("CREATE TABLE IF NOT EXISTS FeatureResult_new ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                    + "inspectionId INTEGER NOT NULL,"
                    + "featureType TEXT,"
                    + "expectedCount INTEGER NOT NULL,"
                    + "detectedCount INTEGER NOT NULL,"
                    + "status TEXT,"
                    + "severity INTEGER NOT NULL,"
                    + "FOREIGN KEY(inspectionId) REFERENCES InspectionRecord(id)"
                    + " ON UPDATE NO ACTION ON DELETE CASCADE)");
            // Rows whose parent is already gone are fragments of a report that no longer
            // exists. Nothing can render them and no digest can verify them.
            db.execSQL("INSERT INTO FeatureResult_new"
                    + " (id, inspectionId, featureType, expectedCount, detectedCount, status, severity)"
                    + " SELECT id, inspectionId, featureType, expectedCount, detectedCount, status,"
                    + "        severity"
                    + " FROM FeatureResult"
                    + " WHERE inspectionId IN (SELECT id FROM InspectionRecord)");
            db.execSQL("DROP TABLE FeatureResult");
            db.execSQL("ALTER TABLE FeatureResult_new RENAME TO FeatureResult");
            db.execSQL("CREATE INDEX IF NOT EXISTS index_FeatureResult_inspectionId"
                    + " ON FeatureResult (inspectionId)");

            // ── SlotResult: same reasoning ─────────────────────────────────────
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
                    + "offsetPct REAL,"
                    + "zone TEXT,"
                    + "FOREIGN KEY(inspectionId) REFERENCES InspectionRecord(id)"
                    + " ON UPDATE NO ACTION ON DELETE CASCADE)");
            db.execSQL("INSERT INTO SlotResult_new"
                    + " (id, inspectionId, slotId, expectedType, foundType, status, x, y,"
                    + "  relX, relY, offsetPct, zone)"
                    + " SELECT id, inspectionId, slotId, expectedType, foundType, status, x, y,"
                    + "        relX, relY, offsetPct, zone"
                    + " FROM SlotResult"
                    + " WHERE inspectionId IN (SELECT id FROM InspectionRecord)");
            db.execSQL("DROP TABLE SlotResult");
            db.execSQL("ALTER TABLE SlotResult_new RENAME TO SlotResult");
            db.execSQL("CREATE INDEX IF NOT EXISTS index_SlotResult_inspectionId"
                    + " ON SlotResult (inspectionId)");

            // ── ClassificationEvent: set null, so nothing is lost ──────────────
            db.execSQL("CREATE TABLE IF NOT EXISTS ClassificationEvent_new ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                    + "partId TEXT,"
                    + "inspectionId INTEGER,"
                    + "timestamp INTEGER NOT NULL,"
                    + "cropPath TEXT,"
                    + "predictedType TEXT,"
                    + "predictedScore REAL NOT NULL,"
                    + "runnerUpType TEXT,"
                    + "runnerUpScore REAL NOT NULL,"
                    + "reason TEXT,"
                    + "similarityThreshold REAL NOT NULL,"
                    + "marginThreshold REAL NOT NULL,"
                    + "encoderId TEXT,"
                    + "confirmedType TEXT,"
                    + "reviewed INTEGER NOT NULL,"
                    + "reviewSource TEXT,"
                    + "FOREIGN KEY(inspectionId) REFERENCES InspectionRecord(id)"
                    + " ON UPDATE NO ACTION ON DELETE SET NULL)");
            // Two cases collapse to the same answer. The -1 sentinel meant "the operator never
            // saved the report"; an id with no surviving parent means the report went away
            // afterwards. Both are "attached to no inspection", and neither is a reason to throw
            // away a human-labelled crop — these rows are what ThresholdTuner fits against, and
            // they are read by partId and reviewed, never by inspectionId.
            db.execSQL("INSERT INTO ClassificationEvent_new"
                    + " (id, partId, inspectionId, timestamp, cropPath, predictedType,"
                    + "  predictedScore, runnerUpType, runnerUpScore, reason,"
                    + "  similarityThreshold, marginThreshold, encoderId, confirmedType,"
                    + "  reviewed, reviewSource)"
                    + " SELECT id, partId,"
                    + "        CASE WHEN inspectionId IN (SELECT id FROM InspectionRecord)"
                    + "             THEN inspectionId ELSE NULL END,"
                    + "        timestamp, cropPath, predictedType, predictedScore, runnerUpType,"
                    + "        runnerUpScore, reason, similarityThreshold, marginThreshold,"
                    + "        encoderId, confirmedType, reviewed, reviewSource"
                    + " FROM ClassificationEvent");
            db.execSQL("DROP TABLE ClassificationEvent");
            db.execSQL("ALTER TABLE ClassificationEvent_new RENAME TO ClassificationEvent");
            db.execSQL("CREATE INDEX IF NOT EXISTS index_ClassificationEvent_inspectionId"
                    + " ON ClassificationEvent (inspectionId)");
            db.execSQL("CREATE INDEX IF NOT EXISTS"
                    + " index_ClassificationEvent_partId_reviewed_timestamp"
                    + " ON ClassificationEvent (partId, reviewed, timestamp)");

            // ── The advisory-notes join ────────────────────────────────────────
            db.execSQL("CREATE INDEX IF NOT EXISTS index_InspectionRecord_partId"
                    + " ON InspectionRecord (partId)");
        }
    };

    /**
     * Introduces products, and the part registry that hangs off them.
     *
     * <p>Until now the part list was assembled at render time from whatever
     * {@code ReferenceProfile} rows happened to exist, plus three hardcoded preset ids. That
     * makes a part something that only exists once it has a photo, which is backwards: an
     * operator wants to register the part first and photograph it second, and wants to see the
     * ones still waiting for a reference.
     *
     * <p>Everything that existed before lands in one product, so nothing disappears from the
     * list on upgrade. The presets are seeded here rather than left hardcoded in the fragment —
     * once parts are rows the operator can rename and regroup, a list half from the database
     * and half from a constant would be half editable.
     *
     * <p>displayName is left null for every backfilled row. Null means "no label set" and the
     * list falls back to the id, which is exactly what was on screen before this ran; writing
     * the id into the label as well would make an unrenamed part indistinguishable from one
     * deliberately renamed to match its id.
     */
    static final Migration MIGRATION_11_12 = new Migration(11, 12) {
        @Override
        public void migrate(@NonNull SupportSQLiteDatabase db) {
            db.execSQL("CREATE TABLE IF NOT EXISTS Product ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                    + "name TEXT,"
                    + "createdAt INTEGER NOT NULL)");
            db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS index_Product_name ON Product (name)");

            db.execSQL("CREATE TABLE IF NOT EXISTS PartProfile ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
                    + "productId INTEGER NOT NULL,"
                    + "partId TEXT,"
                    + "displayName TEXT,"
                    + "createdAt INTEGER NOT NULL,"
                    + "FOREIGN KEY(productId) REFERENCES Product(id)"
                    + " ON UPDATE NO ACTION ON DELETE CASCADE)");
            db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS index_PartProfile_partId"
                    + " ON PartProfile (partId)");
            db.execSQL("CREATE INDEX IF NOT EXISTS index_PartProfile_productId"
                    + " ON PartProfile (productId)");

            long now = System.currentTimeMillis();
            db.execSQL("INSERT INTO Product (name, createdAt) VALUES (?, ?)",
                    new Object[]{DEFAULT_PRODUCT_NAME, now});

            long productId;
            try (Cursor c = db.query("SELECT id FROM Product WHERE name = ?",
                    new Object[]{DEFAULT_PRODUCT_NAME})) {
                if (!c.moveToFirst()) return;
                productId = c.getLong(0);
            }

            // Ordered oldest first so the registry keeps the order the list already showed.
            try (Cursor c = db.query("SELECT DISTINCT partType FROM ReferenceProfile"
                    + " WHERE partType IS NOT NULL AND partType != '' ORDER BY createdAt ASC")) {
                while (c.moveToNext()) {
                    db.execSQL("INSERT OR IGNORE INTO PartProfile"
                                    + " (productId, partId, displayName, createdAt) VALUES (?, ?, NULL, ?)",
                            new Object[]{productId, c.getString(0), now});
                }
            }

            for (String preset : SEED_PART_IDS) {
                db.execSQL("INSERT OR IGNORE INTO PartProfile"
                                + " (productId, partId, displayName, createdAt) VALUES (?, ?, NULL, ?)",
                        new Object[]{productId, preset, now});
            }
        }
    };

    /**
     * Every migration, in one place so that opening the database in a test goes through exactly
     * the list a device does. A test that hand-copied this would keep passing after someone
     * added a migration and forgot it, which is the case worth catching.
     */
    static final Migration[] MIGRATIONS = {
            MIGRATION_3_4, MIGRATION_4_5, MIGRATION_5_6, MIGRATION_6_7,
            MIGRATION_7_8, MIGRATION_8_9, MIGRATION_9_10, MIGRATION_10_11,
            MIGRATION_11_12
    };

    public static AppDatabase getInstance(final Context context) {
        if (INSTANCE == null) {
            synchronized (AppDatabase.class) {
                if (INSTANCE == null) {
                    INSTANCE = Room.databaseBuilder(context.getApplicationContext(),
                                    AppDatabase.class, "shop_quality_inspection_db")
                            .addMigrations(MIGRATIONS)
                            // Scoped to the two unversioned schemas that predate MIGRATION_3_4;
                            // those genuinely have no upgrade path and are only on long-dead dev
                            // installs. It used to be an unscoped fallbackToDestructiveMigration(),
                            // which meant ANY schema mismatch — including hopping between feature
                            // branches on a real device — silently dropped every table and every
                            // recorded inspection with it. Anything from v3 up now fails loudly
                            // instead, which is what you want to find out in testing rather than
                            // after a day of collected data has gone.
                            .fallbackToDestructiveMigrationFrom(true, 1, 2)
                            .build();
                }
            }
        }
        return INSTANCE;
    }
}
