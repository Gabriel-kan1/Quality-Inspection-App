package com.example.data;

import android.content.Context;
import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;

import com.example.data.dao.InspectionDao;
import com.example.data.dao.ReferenceProfileDao;
import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.ReferenceProfile;

@Database(entities = {InspectionRecord.class, FeatureResult.class, ReferenceProfile.class}, version = 1, exportSchema = false)
public abstract class AppDatabase extends RoomDatabase {
    private static volatile AppDatabase INSTANCE;

    public abstract InspectionDao inspectionDao();
    public abstract ReferenceProfileDao referenceProfileDao();

    public static AppDatabase getInstance(final Context context) {
        if (INSTANCE == null) {
            synchronized (AppDatabase.class) {
                if (INSTANCE == null) {
                    INSTANCE = Room.databaseBuilder(context.getApplicationContext(),
                                    AppDatabase.class, "shop_quality_inspection_db")
                            .fallbackToDestructiveMigration()
                            .build();
                }
            }
        }
        return INSTANCE;
    }
}
