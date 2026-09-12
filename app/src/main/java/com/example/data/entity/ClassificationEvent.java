package com.example.data.entity;

import androidx.annotation.Nullable;

import androidx.room.Entity;
import androidx.room.ForeignKey;
import androidx.room.Index;
import androidx.room.PrimaryKey;

/**
 * One logged classification decision — the raw material of the self-learning loop.
 *
 * Every detected crop produces an event, whether or not a human ever looks at it.
 * Events the operator confirms or corrects ({@link #reviewed} = true) become the
 * labelled dataset that {@link com.example.ml.ThresholdTuner} fits per-part
 * thresholds against, and that a future off-device fine-tune would train on.
 */
@Entity(tableName = "ClassificationEvent",
        // SET NULL, not CASCADE, and the difference matters. These rows are not part of the
        // report — they are human-labelled crops, and ThresholdTuner reads them by partId and
        // reviewed without ever looking at inspectionId. Cascading would delete labelled
        // training data as a side effect of tidying up a report, quietly degrading the
        // classifier for that part. The report tables cascade because they genuinely are the
        // report; this one outlives it.
        foreignKeys = @ForeignKey(entity = InspectionRecord.class,
                parentColumns = "id", childColumns = "inspectionId",
                onDelete = ForeignKey.SET_NULL),
        indices = {
                @Index("inspectionId"),
                // The learning loop's hot path, over the table that grows fastest: one row per
                // detection, up to 300 per inspection. Covers getReviewedForPart,
                // getRecentReviewedForPart including its ORDER BY, getUnreviewedCropPaths,
                // countReviewedForPart and deleteStaleForPart.
                @Index({"partId", "reviewed", "timestamp"})
        })
public class ClassificationEvent {

    /** Set on {@link #reviewSource} — how this event came to be reviewed by a human. */
    public static final String SOURCE_UNKNOWN_PROMPT = "unknown_prompt";
    public static final String SOURCE_SPOT_CHECK = "spot_check";
    /**
     * Operator tapped a box on the overlay to correct it, unprompted. Tracked separately
     * because these are self-selected — people tap what looks wrong — so they over-represent
     * errors relative to what the classifier actually produces.
     */
    public static final String SOURCE_OVERLAY_TAP = "overlay_tap";
    /**
     * A detection the {@link com.example.ml.NoiseGate} suppressed, put back in front of the
     * operator to check the gate. Tracked separately so the gate's own error rate can be
     * measured from the history rather than assumed.
     */
    public static final String SOURCE_NOISE_AUDIT = "noise_audit";
    /**
     * Labelled during reference setup, where the operator reviews every crop of the reference
     * photo with the classifier's closest match pre-filled. These are the most carefully
     * given labels in the system and were previously never recorded — the noise gate and the
     * tuner saw only inspection reviews. Tracked separately because one photo yields many
     * correlated labels at once: a study that wants independent samples can leave them out.
     */
    public static final String SOURCE_REFERENCE_SETUP = "reference_setup";

    @PrimaryKey(autoGenerate = true)
    private long id;
    private String partId;
    /**
     * Inspection this belonged to, or null when it belongs to none.
     *
     * <p>Events are logged during the pipeline, before any inspection row exists, and are
     * attached later by {@code ClassificationEventDao.assignInspection} — so "not attached"
     * is a normal, permanent state for an event whose report the operator never saved.
     *
     * <p>This was a {@code long} defaulting to -1. A sentinel cannot survive the foreign key
     * above, because no InspectionRecord has that id, and it should not have to: SQL already
     * has a way to say "no parent". Boxing is what makes the column nullable without needing a
     * SQL default, the same trade {@code SlotResult.offsetPct} makes.
     */
    @Nullable
    private Long inspectionId;
    private long timestamp;
    /** Path to the persisted crop PNG, or null if it couldn't be written. */
    private String cropPath;

    private String predictedType;
    private float predictedScore;
    private String runnerUpType;
    private float runnerUpScore;
    /** {@link com.example.ml.ReferenceLibrary.Reason} name at the time of prediction. */
    private String reason;

    /** Thresholds in force when this prediction was made, so replays stay honest. */
    private float similarityThreshold;
    private float marginThreshold;

    /**
     * {@link com.example.ml.FeatureEmbedder#ENCODER_ID} at the time of prediction.
     *
     * The scores above are only comparable within one embedding space, so replaying a
     * threshold against an event from a different encoder is meaningless — the two
     * encoders don't even use the same numeric range. ThresholdTuner filters on this
     * rather than fitting across a mixed history. Rows written before this column
     * existed carry {@link #ENCODER_LEGACY_UNTAGGED}.
     */
    private String encoderId;

    /** Marks pre-migration rows, which came from the retired CLIP encoder. */
    public static final String ENCODER_LEGACY_UNTAGGED = "legacy-untagged";

    /** Ground truth once a human weighs in: a type key, or "unknown" if they skipped. */
    private String confirmedType;
    private boolean reviewed;
    private String reviewSource;

    public ClassificationEvent() {
    }

    public long getId() { return id; }
    public void setId(long id) { this.id = id; }

    public String getPartId() { return partId; }
    public void setPartId(String partId) { this.partId = partId; }

    @Nullable
    public Long getInspectionId() { return inspectionId; }
    public void setInspectionId(@Nullable Long inspectionId) { this.inspectionId = inspectionId; }

    public long getTimestamp() { return timestamp; }
    public void setTimestamp(long timestamp) { this.timestamp = timestamp; }

    public String getCropPath() { return cropPath; }
    public void setCropPath(String cropPath) { this.cropPath = cropPath; }

    public String getPredictedType() { return predictedType; }
    public void setPredictedType(String predictedType) { this.predictedType = predictedType; }

    public float getPredictedScore() { return predictedScore; }
    public void setPredictedScore(float predictedScore) { this.predictedScore = predictedScore; }

    public String getRunnerUpType() { return runnerUpType; }
    public void setRunnerUpType(String runnerUpType) { this.runnerUpType = runnerUpType; }

    public float getRunnerUpScore() { return runnerUpScore; }
    public void setRunnerUpScore(float runnerUpScore) { this.runnerUpScore = runnerUpScore; }

    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }

    public float getSimilarityThreshold() { return similarityThreshold; }
    public void setSimilarityThreshold(float similarityThreshold) { this.similarityThreshold = similarityThreshold; }

    public float getMarginThreshold() { return marginThreshold; }
    public void setMarginThreshold(float marginThreshold) { this.marginThreshold = marginThreshold; }

    public String getEncoderId() { return encoderId; }
    public void setEncoderId(String encoderId) { this.encoderId = encoderId; }

    public String getConfirmedType() { return confirmedType; }
    public void setConfirmedType(String confirmedType) { this.confirmedType = confirmedType; }

    public boolean isReviewed() { return reviewed; }
    public void setReviewed(boolean reviewed) { this.reviewed = reviewed; }

    public String getReviewSource() { return reviewSource; }
    public void setReviewSource(String reviewSource) { this.reviewSource = reviewSource; }
}
