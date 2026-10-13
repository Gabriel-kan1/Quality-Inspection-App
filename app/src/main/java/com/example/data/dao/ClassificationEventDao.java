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

    /**
     * The most recent {@code limit} human-labelled events for a part.
     *
     * <p>What ThresholdTuner actually needs. The unbounded variant above replays the entire
     * history over a ~2000-point grid on every inspection, so its cost grows without limit as
     * a part accumulates reviews — and the oldest reviews are the least representative of how
     * the part is being photographed now.
     */
    @Query("SELECT * FROM ClassificationEvent WHERE partId = :partId AND reviewed = 1 "
            + "ORDER BY timestamp DESC LIMIT :limit")
    List<ClassificationEvent> getRecentReviewedForPart(String partId, int limit);

    /**
     * Crop paths whose events have not been reviewed yet, so pruning can spare them — the
     * crop is the only evidence a later review could look at.
     */
    @Query("SELECT cropPath FROM ClassificationEvent WHERE partId = :partId "
            + "AND reviewed = 0 AND cropPath IS NOT NULL")
    List<String> getUnreviewedCropPaths(String partId);

    /**
     * Drops a part's events older than {@code cutoff}, keeping unreviewed ones (still
     * actionable) and any tied to a saved inspection (part of a filed report's audit trail).
     *
     * <p>{@code inspectionId} is a primitive defaulting to -1 rather than a nullable column,
     * so "not attached to an inspection" is {@code = -1}, not {@code IS NULL}.
     */
    @Query("DELETE FROM ClassificationEvent WHERE partId = :partId AND timestamp < :cutoff "
            + "AND reviewed = 1 AND inspectionId = -1")
    int deleteStaleForPart(String partId, long cutoff);

    @Query("SELECT COUNT(*) FROM ClassificationEvent WHERE partId = :partId AND reviewed = 1")
    int countReviewedForPart(String partId);

    @Query("SELECT COUNT(*) FROM ClassificationEvent WHERE partId = :partId")
    int countForPart(String partId);

    @Query("SELECT * FROM ClassificationEvent WHERE partId = :partId ORDER BY timestamp DESC")
    List<ClassificationEvent> getAllForPart(String partId);

    /** Every reviewed event across all parts — the export set for off-device fine-tuning. */
    @Query("SELECT * FROM ClassificationEvent WHERE reviewed = 1 ORDER BY timestamp ASC")
    List<ClassificationEvent> getAllReviewed();

    /** Attaches unassigned events to the inspection once the operator saves the report. */
    /** Drops the given rows: an inspection re-run with another method replaces its events. */
    @Query("DELETE FROM ClassificationEvent WHERE id IN (:ids)")
    int deleteByIds(java.util.List<Long> ids);

    @Query("UPDATE ClassificationEvent SET inspectionId = :inspectionId WHERE id IN (:ids)")
    void assignInspection(Long inspectionId, List<Long> ids);
}
