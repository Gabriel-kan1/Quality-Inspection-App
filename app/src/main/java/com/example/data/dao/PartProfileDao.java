package com.example.data.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;

import com.example.data.entity.PartProfile;

import java.util.List;

@Dao
public interface PartProfileDao {

    @Query("SELECT * FROM PartProfile ORDER BY createdAt ASC, id ASC")
    List<PartProfile> getAllPartProfiles();

    @Query("SELECT * FROM PartProfile WHERE productId = :productId"
            + " ORDER BY createdAt ASC, id ASC")
    List<PartProfile> getPartProfilesForProduct(long productId);

    @Query("SELECT * FROM PartProfile WHERE partId = :partId LIMIT 1")
    PartProfile getPartProfileByPartId(String partId);

    @Insert
    long insertPartProfile(PartProfile profile);

    /** Renames the label only. The part id is the key everything else hangs off. */
    @Query("UPDATE PartProfile SET displayName = :displayName WHERE id = :id")
    void renamePartProfile(long id, String displayName);

    @Query("UPDATE PartProfile SET productId = :productId WHERE id = :id")
    void movePartProfile(long id, long productId);

    @Query("DELETE FROM PartProfile WHERE id = :id")
    void deletePartProfile(long id);
}
