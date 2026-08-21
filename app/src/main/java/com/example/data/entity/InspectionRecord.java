package com.example.data.entity;

import androidx.room.Entity;
import androidx.room.Index;
import androidx.room.PrimaryKey;

/**
 * One filed inspection.
 *
 * <p>Rows here are meant to be <em>append-only</em>. A quality record that can be edited in
 * place is not a record, so a re-inspection inserts a fresh row pointing back at the one it
 * replaces via {@link #getSupersedesUuid()} rather than updating it, and {@link #getDigest()}
 * makes a later edit detectable. The one legitimate write after insert is stamping the digest
 * itself, which cannot be computed until the child rows exist — see
 * {@code InspectionDao.sealInspection}.
 */
@Entity(tableName = "InspectionRecord", indices = {
        @Index(value = "uuid", unique = true),
        @Index("canonicalPartId"),
        // The join in InspectionDao.getHistoricalDetectedCounts, which is what builds the
        // report's advisory trend notes, filters on this for every advisory type on every run.
        @Index("partId")
})
public class InspectionRecord {
    @PrimaryKey(autoGenerate = true)
    private long id;
    private String partId;
    private long timestamp;
    // "PASS", "FAIL", or "UNVERIFIED" when the part had no reference image to compare
    // against, so no expected configuration existed. See ResultsFragment.Verdict.
    private String passOrFail;
    private String imageUri;
    private String notes;

    /**
     * Stable identity for this inspection, independent of the autoincrement rowid.
     *
     * <p>The rowid is fine inside one database and useless outside it: it collides across
     * devices, changes on a restore, and says nothing about the record it names. This is what
     * the printed QR encodes, what an export filename carries and what a report is looked up
     * by, so it has to survive leaving the device and coming back.
     */
    private String uuid;

    /**
     * Hex SHA-256 over the canonical serialisation of this record and its child rows, as
     * produced by {@code InspectionExport.canonicalForm}.
     *
     * <p>Null on rows filed before sealing existed, and that is deliberate. The digest's whole
     * claim is "nothing has changed since this was filed"; back-computing one during the
     * migration would compute it over whatever the row holds <em>now</em> and assert a
     * guarantee that was never in force. An unsealed row says so instead.
     */
    private String digest;

    /**
     * Which version of the export schema this record was written against, so an exported file
     * can be read by a later build without guessing.
     *
     * <p>Boxed for the same reason {@code SlotResult.offsetPct} is: a nullable column needs no
     * SQL default, which is what lets the migration be a plain ALTER. Null means the row
     * predates sealing, matching a null {@link #digest}.
     */
    private Integer schemaVersion;

    /**
     * The {@link #uuid} of the inspection this one replaces, or null for a first inspection.
     * Re-inspecting a part appends rather than overwrites, so the chain stays readable.
     */
    private String supersedesUuid;

    /**
     * Who ran the inspection. A device or shift identifier is enough — this is not a login
     * system — but a quality record that cannot say who signed it off is incomplete.
     */
    private String operator;

    /**
     * {@code partId} reduced by {@code PartFiles.safeName}, which is the form every on-disk
     * path for this part is actually built from.
     *
     * <p>Stored rather than derived because it is the only way to <em>see</em> the collision it
     * represents: the database keeps part ids raw, so "A/B" and "A_B" are two distinct records
     * that nonetheless share one layout file, one prototype library, one threshold file and one
     * crop folder. Grouping by this column is what makes that answerable.
     */
    private String canonicalPartId;

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

    public String getUuid() {
        return uuid;
    }

    public void setUuid(String uuid) {
        this.uuid = uuid;
    }

    public String getDigest() {
        return digest;
    }

    public void setDigest(String digest) {
        this.digest = digest;
    }

    public Integer getSchemaVersion() {
        return schemaVersion;
    }

    public void setSchemaVersion(Integer schemaVersion) {
        this.schemaVersion = schemaVersion;
    }

    public String getSupersedesUuid() {
        return supersedesUuid;
    }

    public void setSupersedesUuid(String supersedesUuid) {
        this.supersedesUuid = supersedesUuid;
    }

    public String getOperator() {
        return operator;
    }

    public void setOperator(String operator) {
        this.operator = operator;
    }

    public String getCanonicalPartId() {
        return canonicalPartId;
    }

    public void setCanonicalPartId(String canonicalPartId) {
        this.canonicalPartId = canonicalPartId;
    }
}
