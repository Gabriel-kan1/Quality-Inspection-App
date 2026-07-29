package com.example.data.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;
import androidx.room.Update;

import com.example.data.entity.ClassificationEvent;

import java.util.List;

@Dao
public interface ClassificationEventDao {

    @Insert
    long insert(ClassificationEvent event);

    @Update
    void update(ClassificationEvent event);

    @Query("SELECT * FROM ClassificationEvent WHERE id = :id LIMIT 1")
    ClassificationEvent getById(long id);

    /** Records a human's verdict on a previously logged prediction. */
    @Query("UPDATE ClassificationEvent SET confirmedType = :confirmedType, reviewed = 1, "
            + "reviewSource = :reviewSource WHERE id = :id")
    void recordReview(long id, String confirmedType, String reviewSource);

    /** Human-labelled events for a part — the training/tuning set. */
    @Query("SELECT * FROM ClassificationEvent WHERE partId = :partId AND reviewed = 1 ORDER BY timestamp ASC")
    List<ClassificationEvent> getReviewedForPart(String partId);

    @Query("SELECT COUNT(*) FROM ClassificationEvent WHERE partId = :partId AND reviewed = 1")
    int countReviewedForPart(String partId);

    @Query("SELECT COUNT(*) FROM ClassificationEvent WHERE partId = :partId")
    int countForPart(String partId);

    @Query("SELECT * FROM ClassificationEvent WHERE partId = :partId ORDER BY timestamp DESC")
    List<ClassificationEvent> getAllForPart(String partId);

    /** Every reviewed event across all parts — the export set for off-device fine-tuning. */
    @Query("SELECT * FROM ClassificationEvent WHERE reviewed = 1 ORDER BY timestamp ASC")
    List<ClassificationEvent> getAllReviewed();

    /** Attaches orphan events to the inspection once the operator saves the report. */
    @Query("UPDATE ClassificationEvent SET inspectionId = :inspectionId WHERE id IN (:ids)")
    void assignInspection(long inspectionId, List<Long> ids);
}
