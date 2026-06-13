package com.example.data.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;

import com.example.data.entity.ReferenceProfile;

import java.util.List;

@Dao
public interface ReferenceProfileDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    long insertProfile(ReferenceProfile profile);

    @Query("SELECT * FROM ReferenceProfile WHERE partType = :partType LIMIT 1")
    ReferenceProfile getProfileByPartType(String partType);

    @Query("SELECT * FROM ReferenceProfile ORDER BY createdAt DESC")
    List<ReferenceProfile> getAllProfiles();
}
