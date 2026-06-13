package com.example.data.entity;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "ReferenceProfile")
public class ReferenceProfile {
    @PrimaryKey(autoGenerate = true)
    private long id;
    private String partType; // e.g., "PART_A", "PART_B" or barcode/ID family
    private String imageUri;
    private long createdAt;

    public ReferenceProfile() {
    }

    public ReferenceProfile(String partType, String imageUri, long createdAt) {
        this.partType = partType;
        this.imageUri = imageUri;
        this.createdAt = createdAt;
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getPartType() {
        return partType;
    }

    public void setPartType(String partType) {
        this.partType = partType;
    }

    public String getImageUri() {
        return imageUri;
    }

    public void setImageUri(String imageUri) {
        this.imageUri = imageUri;
    }

    public long getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(long createdAt) {
        this.createdAt = createdAt;
    }
}
