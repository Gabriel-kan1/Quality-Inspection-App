package com.example.data.entity;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "ReferenceProfile")
public class ReferenceProfile {

    /**
     * Stand-in {@code imageUri} for a part registered without a reference photo yet.
     *
     * <p>Public because the check "is this a real reference?" was written out by hand at five
     * call sites, each repeating both the sentinel string and the null guard around it. A
     * sixth that forgot either half would silently treat the placeholder as a photo and try
     * to decode it. Prefer {@link #hasRealImage()} over comparing against this directly.
     */
    public static final String PLACEHOLDER_URI = "dummy_placeholder";

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

    /**
     * True when this profile points at an actual reference photo, rather than being absent
     * or still holding {@link #PLACEHOLDER_URI}.
     */
    public boolean hasRealImage() {
        return imageUri != null && !imageUri.isEmpty() && !PLACEHOLDER_URI.equals(imageUri);
    }

    public long getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(long createdAt) {
        this.createdAt = createdAt;
    }
}
