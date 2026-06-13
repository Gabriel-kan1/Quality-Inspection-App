package com.example.data.entity;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "InspectionRecord")
public class InspectionRecord {
    @PrimaryKey(autoGenerate = true)
    private long id;
    private String partId;
    private long timestamp;
    private String passOrFail; // "PASS" or "FAIL"
    private String imageUri;
    private String notes;

    public InspectionRecord() {
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getPartId() {
        return partId;
    }

    public void setPartId(String partId) {
        this.partId = partId;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public String getPassOrFail() {
        return passOrFail;
    }

    public void setPassOrFail(String passOrFail) {
        this.passOrFail = passOrFail;
    }

    public String getImageUri() {
        return imageUri;
    }

    public void setImageUri(String imageUri) {
        this.imageUri = imageUri;
    }

    public String getNotes() {
        return notes;
    }

    public void setNotes(String notes) {
        this.notes = notes;
    }
}
