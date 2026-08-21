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

    /**
     * The unique index on partType means at most one row can match. The ordering is kept as a
     * backstop so that if a duplicate ever reappears the newest real photo wins rather than
     * whichever row the query planner reaches first — the previous unordered LIMIT 1 returned
     * the oldest, which is how a stale placeholder could shadow a saved reference indefinitely.
     */
    @Query("SELECT * FROM ReferenceProfile WHERE partType = :partType"
            + " ORDER BY CASE WHEN imageUri = 'dummy_placeholder' THEN 1 ELSE 0 END,"
            + " createdAt DESC LIMIT 1")
    ReferenceProfile getProfileByPartType(String partType);

    @Query("SELECT * FROM ReferenceProfile ORDER BY createdAt DESC")
    List<ReferenceProfile> getAllProfiles();

    /**
     * Drops the reference photo for a part.
     *
     * <p>Needed when a part is removed from the registry: the part list adopts any part that
     * has a reference but no registry row, so leaving this behind would make a deleted part
     * reappear on the next refresh.
     */
    @Query("DELETE FROM ReferenceProfile WHERE partType = :partType")
    void deleteByPartType(String partType);
}
