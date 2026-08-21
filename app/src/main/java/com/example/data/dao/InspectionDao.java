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

    /**
     * The inspection a printed QR or an exported file names.
     *
     * <p>This is the lookup that matters once a report has left the device: the rowid
     * {@link #getInspectionById} takes is only meaningful inside one database, whereas the
     * uuid is what was actually printed.
     */
    @Query("SELECT * FROM InspectionRecord WHERE uuid = :uuid LIMIT 1")
    InspectionRecord getInspectionByUuid(String uuid);

    /**
     * Stamps the integrity seal on an inspection that has just been filed.
     *
     * <p>The one update a filed record is allowed to take, and only because the digest covers
     * the child rows, which do not exist until after the parent is inserted. Everything else
     * about a filed inspection is append-only — a re-inspection inserts a new row carrying
     * {@code supersedesUuid} instead of rewriting this one.
     */
    @Query("UPDATE InspectionRecord SET digest = :digest, schemaVersion = :schemaVersion"
            + " WHERE id = :id")
    void sealInspection(long id, String digest, int schemaVersion);

    /**
     * Every inspection of a part, newest first, keyed by the sanitised id rather than the raw
     * one — which is to say, by the identity the part's files on disk actually have.
     *
     * <p>Two records whose raw ids differ only in characters the sanitiser collapses ("A/B"
     * and "A_B") share a layout, a prototype library, a threshold file and a crop folder.
     * That is invisible when grouping by partId and obvious here.
     */
    @Query("SELECT * FROM InspectionRecord WHERE canonicalPartId = :canonicalPartId"
            + " ORDER BY timestamp DESC")
    List<InspectionRecord> getInspectionsForCanonicalPart(String canonicalPartId);

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

    /**
     * Removes an inspection and, by the foreign keys on the child tables, everything that was
     * part of it.
     *
     * <p>The two companion methods that used to sit here — one per child table — existed only
     * to emulate that cascade by hand, and calling one without the other left orphans nothing
     * would object to. The schema does it now, so this is the single correct way.
     *
     * <p>Classification events are the exception and survive with a null inspectionId: they are
     * labelled training data, not part of the report. See {@code ClassificationEvent}.
     */
    @Query("DELETE FROM InspectionRecord WHERE id = :id")
    void deleteInspection(long id);
}
