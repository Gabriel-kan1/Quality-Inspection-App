package com.example.data.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;
import androidx.room.Transaction;

import com.example.data.entity.InspectionRecord;
import com.example.data.entity.FeatureResult;
import com.example.data.entity.SlotResult;

import java.util.List;

@Dao
public interface InspectionDao {
    @Insert
    long insertInspection(InspectionRecord record);

    @Insert
    void insertFeatureResults(List<FeatureResult> results);

    @Query("SELECT * FROM InspectionRecord ORDER BY timestamp DESC")
    List<InspectionRecord> getAllInspections();

    @Query("SELECT * FROM InspectionRecord WHERE partId LIKE :searchPattern ORDER BY timestamp DESC")
    List<InspectionRecord> searchInspections(String searchPattern);

    @Query("SELECT * FROM InspectionRecord WHERE id = :id LIMIT 1")
    InspectionRecord getInspectionById(long id);

    @Query("SELECT * FROM FeatureResult WHERE inspectionId = :inspectionId")
    List<FeatureResult> getFeatureResultsForInspection(long inspectionId);

    /**
     * Detected counts for one feature type on earlier inspections of the same part, newest
     * first — the baseline the report's warning notes compare this run against.
     *
     * Scoped to partId because the counts are only comparable within a part: three scratches
     * on a small bracket and three on a chassis panel are not the same observation. The
     * current inspection is excluded by id so it cannot pull its own baseline toward itself.
     */
    @Query("SELECT fr.detectedCount FROM FeatureResult fr"
            + " INNER JOIN InspectionRecord ir ON ir.id = fr.inspectionId"
            + " WHERE ir.partId = :partId AND fr.featureType = :featureType"
            + " AND fr.inspectionId != :excludeInspectionId"
            + " ORDER BY ir.timestamp DESC LIMIT :limit")
    List<Integer> getHistoricalDetectedCounts(String partId, String featureType,
                                              long excludeInspectionId, int limit);

    @Insert
    void insertSlotResults(List<SlotResult> results);

    @Query("SELECT * FROM SlotResult WHERE inspectionId = :inspectionId ORDER BY slotId")
    List<SlotResult> getSlotResultsForInspection(long inspectionId);

    @Query("DELETE FROM InspectionRecord WHERE id = :id")
    void deleteInspection(long id);

    @Query("DELETE FROM FeatureResult WHERE inspectionId = :inspectionId")
    void deleteFeatureResultsForInspection(long inspectionId);

    @Query("DELETE FROM SlotResult WHERE inspectionId = :inspectionId")
    void deleteSlotResultsForInspection(long inspectionId);
}
