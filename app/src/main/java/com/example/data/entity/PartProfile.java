package com.example.data.entity;

import androidx.annotation.Nullable;
import androidx.room.Entity;
import androidx.room.ForeignKey;
import androidx.room.Index;
import androidx.room.PrimaryKey;

/**
 * A part the operator has registered, and which product it belongs to.
 *
 * <p>Separate from {@link ReferenceProfile}, which is the reference <em>photo</em> for a part
 * and only exists once one has been taken. This is the registry: a part appears in the list as
 * soon as it is added, before it has a photo, a layout or a single inspection.
 *
 * <h2>Why partId is not the thing you rename</h2>
 *
 * <p>{@link #partId} is the key the whole app is built on — {@code ReferenceProfile.partType},
 * the layout, prototype-library and threshold files (all named from {@code PartFiles.safeName}),
 * the training-crop folder, every {@code ClassificationEvent} and every {@code InspectionRecord}.
 * Changing it would orphan all of that at once, silently: the part would simply appear to have
 * lost its reference and its history.
 *
 * <p>So renaming edits {@link #displayName} instead, which nothing else reads. The id stays put
 * and stays correct; the label is free to be whatever makes sense on the shop floor. A null
 * display name means "no label set", and the list falls back to showing the id.
 */
@Entity(tableName = "PartProfile",
        foreignKeys = @ForeignKey(entity = Product.class,
                parentColumns = "id", childColumns = "productId",
                onDelete = ForeignKey.CASCADE),
        indices = {
                // One registry row per part: adding a part that already exists moves it rather
                // than duplicating it into two products.
                @Index(value = "partId", unique = true),
                @Index("productId")
        })
public class PartProfile {

    @PrimaryKey(autoGenerate = true)
    private long id;
    private long productId;

    /** The real identifier. Immutable once created — see the class comment. */
    private String partId;

    /** Operator-facing label, or null to show {@link #partId}. */
    @Nullable
    private String displayName;

    private long createdAt;

    public PartProfile() { }

    public PartProfile(long productId, String partId, @Nullable String displayName,
                       long createdAt) {
        this.productId = productId;
        this.partId = partId;
        this.displayName = displayName;
        this.createdAt = createdAt;
    }

    public long getId() { return id; }
    public void setId(long id) { this.id = id; }

    public long getProductId() { return productId; }
    public void setProductId(long productId) { this.productId = productId; }

    public String getPartId() { return partId; }
    public void setPartId(String partId) { this.partId = partId; }

    @Nullable
    public String getDisplayName() { return displayName; }
    public void setDisplayName(@Nullable String displayName) { this.displayName = displayName; }

    public long getCreatedAt() { return createdAt; }
    public void setCreatedAt(long createdAt) { this.createdAt = createdAt; }

    /** What the list shows: the label if one was set, otherwise the id itself. */
    public String label() {
        return displayName == null || displayName.trim().isEmpty() ? partId : displayName;
    }
}
