package com.example.data.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;
import androidx.room.Transaction;

import com.example.data.entity.InspectionRecord;
import com.example.data.entity.FeatureResult;

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

    @Query("DELETE FROM InspectionRecord WHERE id = :id")
    void deleteInspection(long id);

    @Query("DELETE FROM FeatureResult WHERE inspectionId = :inspectionId")
    void deleteFeatureResultsForInspection(long inspectionId);
}
