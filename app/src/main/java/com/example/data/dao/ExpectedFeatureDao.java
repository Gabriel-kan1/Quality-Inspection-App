package com.example.data.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;
import androidx.room.Transaction;

import com.example.data.entity.ExpectedFeature;

import java.util.List;

@Dao
public interface ExpectedFeatureDao {

    /** The part's declared bill of features, in the order they were entered. */
    @Query("SELECT * FROM ExpectedFeature WHERE partId = :partId ORDER BY id ASC")
    List<ExpectedFeature> getForPart(String partId);

    @Insert
    long insert(ExpectedFeature feature);

    /** Drops a part's declared counts; called when the part itself is removed. */
    @Query("DELETE FROM ExpectedFeature WHERE partId = :partId")
    void deleteForPart(String partId);

    /**
     * Replaces the part's declaration wholesale. Delete-then-insert in one transaction rather
     * than an upsert, so a type the operator zeroed out disappears instead of lingering at its
     * old count.
     */
    @Transaction
    default void replaceForPart(String partId, List<ExpectedFeature> features) {
        deleteForPart(partId);
        for (ExpectedFeature f : features) insert(f);
    }
}
