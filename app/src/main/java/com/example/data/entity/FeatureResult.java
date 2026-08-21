package com.example.data.entity;

import androidx.room.Entity;
import androidx.room.ForeignKey;
import androidx.room.Index;
import androidx.room.PrimaryKey;

/**
 * Deleted with its inspection. A per-type count with no inspection attached is not data, it is
 * debris — and the report digest is computed over these rows, so an orphan could never be
 * verified against anything.
 */
@Entity(tableName = "FeatureResult",
        foreignKeys = @ForeignKey(entity = InspectionRecord.class,
                parentColumns = "id", childColumns = "inspectionId",
                onDelete = ForeignKey.CASCADE),
        indices = {@Index("inspectionId")})
public class FeatureResult {
    @PrimaryKey(autoGenerate = true)
    private long id;
    private long inspectionId;
    private String featureType; // e.g., "Bolt", "Hole", "Screw", "Rivet", "Stud"
    private int expectedCount;
    private int detectedCount;
    private String status; // "PASS", "MISSING", "EXTRA"
    /**
     * How this feature type is reported, as a {@code FeatureTypeRegistry.Severity} level
     * (0 ignored, 1 advisory, 2 critical). Snapshotted at save time rather than looked up
     * when the report is opened: the registry is global and editable, so a later change to
     * a type must not silently rewrite what an already-filed inspection reported.
     */
    private int severity = 2;

    public FeatureResult() {
    }

    public FeatureResult(long inspectionId, String featureType, int expectedCount, int detectedCount, String status, int severity) {
        this.inspectionId = inspectionId;
        this.featureType = featureType;
        this.expectedCount = expectedCount;
        this.detectedCount = detectedCount;
        this.status = status;
        this.severity = severity;
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public long getInspectionId() {
        return inspectionId;
    }

    public void setInspectionId(long inspectionId) {
        this.inspectionId = inspectionId;
    }

    public String getFeatureType() {
        return featureType;
    }

    public void setFeatureType(String featureType) {
        this.featureType = featureType;
    }

    public int getExpectedCount() {
        return expectedCount;
    }

    public void setExpectedCount(int expectedCount) {
        this.expectedCount = expectedCount;
    }

    public int getDetectedCount() {
        return detectedCount;
    }

    public void setDetectedCount(int detectedCount) {
        this.detectedCount = detectedCount;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public int getSeverity() {
        return severity;
    }

    public void setSeverity(int severity) {
        this.severity = severity;
    }
}
