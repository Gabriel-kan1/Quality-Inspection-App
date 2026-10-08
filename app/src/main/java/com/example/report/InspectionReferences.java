package com.example.report;

import android.content.Context;
import android.content.SharedPreferences;

/**
 * Which reference photo each filed inspection was compared against, by inspection uuid.
 *
 * <p>A part keeps one reference row, replaced whenever the operator sets a new one, so the
 * database alone cannot say which photo an older report was judged by. Kept beside the database
 * rather than in it: a column would mean a schema migration for a display detail, and this map
 * is not part of the sealed record.
 */
public final class InspectionReferences {

    private static final String PREFS = "inspection_references";
    /** Stored for an inspection that had no reference to compare against. */
    private static final String NONE = "";

    private InspectionReferences() {}

    /** Notes the reference photo an inspection is being filed against; null when it had none. */
    public static void record(Context context, String inspectionUuid, String referenceUri) {
        if (inspectionUuid == null) return;
        prefs(context).edit()
                .putString(inspectionUuid, referenceUri == null ? NONE : referenceUri)
                .apply();
    }

    /**
     * The reference photo an inspection was compared against: its URI, an empty string when
     * it had none, or null when nothing was recorded — reports filed before this existed.
     */
    public static String recorded(Context context, String inspectionUuid) {
        if (inspectionUuid == null) return null;
        return prefs(context).getString(inspectionUuid, null);
    }

    private static SharedPreferences prefs(Context context) {
        return context.getApplicationContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }
}
