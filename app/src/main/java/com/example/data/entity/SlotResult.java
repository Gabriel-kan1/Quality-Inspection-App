package com.example.data.entity;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

/**
 * One expected position on a part and what the inspection found there.
 *
 * Separate from {@link FeatureResult} because the grain differs: a FeatureResult is one row
 * per feature <em>type</em> ("3 bolt heads expected, 2 found"), whereas localisation is one
 * row per <em>instance</em> ("the bolt head at upper-right is the missing one"). Folding
 * these together would mean either losing per-position detail or duplicating the type totals.
 *
 * Only critical-severity types get rows here — see {@code FeatureLocalisation}.
 */
@Entity(tableName = "SlotResult")
public class SlotResult {
    @PrimaryKey(autoGenerate = true)
    private long id;
    private long inspectionId;
    /** Stable slot number within the part's layout, as shown to the operator. */
    private int slotId;
    /** Display name of the type this position expects. */
    private String expectedType;
    /** Display name of the type actually found, or null when the position was empty. */
    private String foundType;
    /** {@code FeatureLocalisation.Status} name: PRESENT, MISSING or WRONG_TYPE. */
    private String status;
    /** Position within the reference photo's frame, fractional (0-1). Kept as recorded. */
    private float x;
    private float y;
    /**
     * Position relative to the part — 0 at the top-left of the box enclosing every detected
     * feature, 1 at its bottom-right. This is what the report shows, because frame
     * coordinates move with however much background the operator framed.
     *
     * Inspections filed before v8 had no envelope normalisation, so their rows carry the
     * frame values here; that is what those reports displayed.
     */
    private float relX;
    private float relY;
    /**
     * For a DISPLACED finding, how far the detection sits from its position as a fraction of
     * the part's extent. Null for every other status — boxed rather than a sentinel so the
     * column can be added without rebuilding the table, and so "not applicable" is not
     * confusable with "zero offset".
     */
    private Float offsetPct;
    /** Human-readable region, e.g. "upper right" — denormalised so the report needs no geometry. */
    private String zone;

    public SlotResult() {
    }

    public SlotResult(long inspectionId, int slotId, String expectedType, String foundType,
                      String status, float x, float y, float relX, float relY,
                      Float offsetPct, String zone) {
        this.inspectionId = inspectionId;
        this.slotId = slotId;
        this.expectedType = expectedType;
        this.foundType = foundType;
        this.status = status;
        this.x = x;
        this.y = y;
        this.relX = relX;
        this.relY = relY;
        this.offsetPct = offsetPct;
        this.zone = zone;
    }

    public long getId() { return id; }
    public void setId(long id) { this.id = id; }

    public long getInspectionId() { return inspectionId; }
    public void setInspectionId(long inspectionId) { this.inspectionId = inspectionId; }

    public int getSlotId() { return slotId; }
    public void setSlotId(int slotId) { this.slotId = slotId; }

    public String getExpectedType() { return expectedType; }
    public void setExpectedType(String expectedType) { this.expectedType = expectedType; }

    public String getFoundType() { return foundType; }
    public void setFoundType(String foundType) { this.foundType = foundType; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public float getX() { return x; }
    public void setX(float x) { this.x = x; }

    public float getY() { return y; }
    public void setY(float y) { this.y = y; }

    public float getRelX() { return relX; }
    public void setRelX(float relX) { this.relX = relX; }

    public float getRelY() { return relY; }
    public void setRelY(float relY) { this.relY = relY; }

    public Float getOffsetPct() { return offsetPct; }
    public void setOffsetPct(Float offsetPct) { this.offsetPct = offsetPct; }

    public String getZone() { return zone; }
    public void setZone(String zone) { this.zone = zone; }
}
