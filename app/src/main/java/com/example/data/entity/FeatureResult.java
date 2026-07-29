package com.example.data.entity;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "FeatureResult")
public class FeatureResult {
    @PrimaryKey(autoGenerate = true)
    private long id;
    private long inspectionId;
    private String featureType; // e.g., "Bolt", "Hole", "Screw", "Rivet", "Stud"
    private int expectedCount;
    private int detectedCount;
    private String status; // "PASS", "MISSING", "EXTRA"
    /** Whether this feature type should appear in the exported report; set per-type by the user via FeatureTypeRegistry. */
    private boolean critical = true;

    public FeatureResult() {
    }

    public FeatureResult(long inspectionId, String featureType, int expectedCount, int detectedCount, String status, boolean critical) {
        this.inspectionId = inspectionId;
        this.featureType = featureType;
        this.expectedCount = expectedCount;
        this.detectedCount = detectedCount;
        this.status = status;
        this.critical = critical;
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

    public boolean isCritical() {
        return critical;
    }

    public void setCritical(boolean critical) {
        this.critical = critical;
    }
}
