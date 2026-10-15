package com.example.report;

import android.content.Context;
import android.content.SharedPreferences;

import com.example.ui.view.ChassisOverlayView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * Where each defective position sat in the inspection photo, by inspection uuid, so the report
 * can circle it the way the results screen did.
 *
 * <p>{@code SlotResult} keeps the slot's position in the <em>reference</em> frame and relative to
 * the part, never its projection into the inspection photo — that came from the alignment, which
 * is not persisted. Kept beside the database for the same reason as
 * {@link InspectionReferences}: a column would mean a migration for a display detail, and these
 * markers are not part of the sealed record.
 */
public final class InspectionDefectMarkers {

    private static final String PREFS = "inspection_defect_markers";

    private InspectionDefectMarkers() {}

    /**
     * Notes the markers the results screen drew for an inspection being filed. An empty list is
     * stored too, so "nothing to circle" is distinguishable from "filed before this existed".
     */
    public static void record(Context context, String inspectionUuid,
                              List<ChassisOverlayView.ExpectedSlot> markers) {
        if (inspectionUuid == null) return;
        JSONArray array = new JSONArray();
        try {
            for (ChassisOverlayView.ExpectedSlot m : markers) {
                JSONObject o = new JSONObject()
                        .put("n", m.slotId)
                        .put("x", (double) m.x)
                        .put("y", (double) m.y)
                        .put("missing", m.missing);
                if (m.label != null) o.put("label", m.label);
                array.put(o);
            }
        } catch (JSONException e) {
            return;
        }
        prefs(context).edit().putString(inspectionUuid, array.toString()).apply();
    }

    /**
     * The markers recorded for an inspection, or null when none were — reports filed before
     * this existed, which make no claim about where their defects were in the photo.
     */
    public static List<ChassisOverlayView.ExpectedSlot> recorded(Context context,
                                                                  String inspectionUuid) {
        if (inspectionUuid == null) return null;
        String json = prefs(context).getString(inspectionUuid, null);
        if (json == null) return null;
        List<ChassisOverlayView.ExpectedSlot> markers = new ArrayList<>();
        try {
            JSONArray array = new JSONArray(json);
            for (int i = 0; i < array.length(); i++) {
                JSONObject o = array.getJSONObject(i);
                markers.add(new ChassisOverlayView.ExpectedSlot(o.getInt("n"),
                        (float) o.getDouble("x"), (float) o.getDouble("y"),
                        o.getBoolean("missing"), o.optString("label", null)));
            }
        } catch (JSONException e) {
            return null;
        }
        return markers;
    }

    private static SharedPreferences prefs(Context context) {
        return context.getApplicationContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }
}
