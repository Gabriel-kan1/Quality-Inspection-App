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

@Database(entities = {InspectionRecord.class, FeatureResult.class, ReferenceProfile.class,
        ClassificationEvent.class}, version = 5, exportSchema = false)
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

    public static AppDatabase getInstance(final Context context) {
        if (INSTANCE == null) {
            synchronized (AppDatabase.class) {
                if (INSTANCE == null) {
                    INSTANCE = Room.databaseBuilder(context.getApplicationContext(),
                                    AppDatabase.class, "shop_quality_inspection_db")
                            .addMigrations(MIGRATION_3_4, MIGRATION_4_5)
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
