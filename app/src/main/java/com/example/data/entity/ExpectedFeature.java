package com.example.data.entity;

import androidx.room.Entity;
import androidx.room.Index;
import androidx.room.PrimaryKey;

/**
 * How many of one feature type a part is supposed to have, as declared by the operator when
 * the reference was confirmed.
 *
 * <p>These rows are the expected side of every inspection's count comparison. They used to be
 * derived by running edge detection over the reference photograph and having the operator
 * label each box, which made "what should be on this part" depend on which edge method ran,
 * on the lighting in the photo, and on a dozen crops being labelled correctly. A stated count
 * depends on none of that, and it means the reference photograph is never processed at all —
 * only the captured image is.
 *
 * <p>Keyed by part id rather than by reference photo, like the layout and library files, so the
 * counts survive a re-photograph and can exist before a part has a photo (the seeded presets
 * ship with theirs). One row per (part, type); {@code count} is always positive — a type with
 * nothing expected simply has no row.
 */
@Entity(tableName = "ExpectedFeature",
        indices = {
                @Index(value = {"partId", "featureType"}, unique = true),
                @Index("partId")
        })
public class ExpectedFeature {

    @PrimaryKey(autoGenerate = true)
    private long id;
    private String partId;
    private String featureType;
    private int count;

    public ExpectedFeature() { }

    public ExpectedFeature(String partId, String featureType, int count) {
        this.partId = partId;
        this.featureType = featureType;
        this.count = count;
    }

    public long getId() { return id; }
    public void setId(long id) { this.id = id; }

    public String getPartId() { return partId; }
    public void setPartId(String partId) { this.partId = partId; }

    public String getFeatureType() { return featureType; }
    public void setFeatureType(String featureType) { this.featureType = featureType; }

    public int getCount() { return count; }
    public void setCount(int count) { this.count = count; }
}
