package com.example.ml;

import android.content.Context;
import android.util.Log;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * User-editable catalog of feature types: display name + how seriously the type is
 * treated, on the {@link Severity} scale.
 *
 * Seeded once from {@link ReferenceLibrary#FEATURE_TYPES} (all critical by
 * default), then persisted to filesDir/feature_types.json. New types are added
 * on-device when the user labels an "unknown" detection during inspection —
 * severity is always an explicit user choice, never inferred from the name.
 */
public final class FeatureTypeRegistry {

    private static final String TAG = "FeatureTypeRegistry";
    private static final String FILE = "feature_types.json";

    /**
     * How a feature type is treated once an inspection is filed. A scale rather than a
     * boolean because the three behaviours are genuinely different, not two: a missing
     * bolt is a defect, an unusual number of scratches is a trend worth a note, and a
     * patch of bare surface is neither.
     *
     * The ordinal is the persisted form, so values must keep their {@link #level} even if
     * the enum is reordered or extended. Extra stops can be inserted between the ends
     * later; the report only branches on the three behaviours below.
     */
    public enum Severity {
        /**
         * Tracked so the classifier keeps learning it, but absent from the report and from
         * the pass/fail verdict. For the catch-all types an operator defines to stop the
         * review queue asking about background — bare surface, shadow, panel texture.
         */
        IGNORED(0, "Ignored", "Learned, but never reported and never fails a part."),
        /**
         * Excluded from the quantitative table, summarised in the report's warning notes
         * when this run's count sits above the part's historical baseline. For cosmetic
         * findings an operator wants trended rather than counted — scratches, rust, scuffs.
         */
        ADVISORY(1, "Advisory", "Trend note only — flagged when it appears more than usual."),
        /**
         * Counted in the quantitative report and compared against the reference to decide
         * PASS/FAIL. The default for anything unclassified, so nothing drops out silently.
         */
        CRITICAL(2, "Critical", "Counted in the quantitative report; drives pass/fail.");

        public final int level;
        public final String label;
        public final String blurb;

        Severity(int level, String label, String blurb) {
            this.level = level;
            this.label = label;
            this.blurb = blurb;
        }

        /** Highest stop on the scale; the slider in the label dialog runs 0..{@code maxLevel()}. */
        public static int maxLevel() {
            return CRITICAL.level;
        }

        /** Clamps rather than throws: an out-of-range stored level is a corrupt file, not a crash. */
        public static Severity fromLevel(int level) {
            for (Severity s : values()) {
                if (s.level == level) return s;
            }
            return level > CRITICAL.level ? CRITICAL : IGNORED;
        }
    }

    public static final class TypeInfo {
        public final String key;
        public final String displayName;
        public final Severity severity;

        public TypeInfo(String key, String displayName, Severity severity) {
            this.key = key;
            this.displayName = displayName;
            this.severity = severity == null ? Severity.CRITICAL : severity;
        }
    }

    private FeatureTypeRegistry() {}

    /** All known types (built-in + user-added), in insertion order. */
    public static synchronized List<TypeInfo> getAll(Context context) {
        return new ArrayList<>(load(context).values());
    }

    public static synchronized TypeInfo get(Context context, String key) {
        if (key == null) return null;
        return load(context).get(key);
    }

    /**
     * Defaults to {@link Severity#CRITICAL} for anything not explicitly registered, so a
     * type nobody has classified yet cannot silently drop out of a report.
     */
    public static synchronized Severity severityOf(Context context, String key) {
        TypeInfo t = get(context, key);
        return t == null ? Severity.CRITICAL : t.severity;
    }

    /** Convenience for the pass/fail and quantitative-table paths, which only care about the top stop. */
    public static synchronized boolean isCritical(Context context, String key) {
        return severityOf(context, key) == Severity.CRITICAL;
    }

    public static synchronized String displayName(Context context, String key) {
        TypeInfo t = get(context, key);
        if (t != null) return t.displayName;
        return ReferenceLibrary.displayName(key);
    }

    /** Adds a new type or updates an existing one's display name/severity. */
    public static synchronized void upsert(Context context, String key, String displayName, Severity severity) {
        if (key == null || key.isEmpty()) return;
        Map<String, TypeInfo> all = load(context);
        all.put(key, new TypeInfo(key, displayName, severity));
        persist(context, all);
    }

    /** Turns free-text typed by a user into a stable lookup key (lowercase, underscores). */
    public static String keyFor(String rawName) {
        if (rawName == null) return "";
        String key = rawName.trim().toLowerCase(java.util.Locale.US).replaceAll("[^a-z0-9]+", "_");
        key = key.replaceAll("^_+|_+$", "");
        return key;
    }

    private static Map<String, TypeInfo> load(Context context) {
        File f = file(context);
        Map<String, TypeInfo> map = new LinkedHashMap<>();
        if (f.exists()) {
            try (InputStream is = new java.io.FileInputStream(f)) {
                String json = PartFiles.readAllText(is);
                JSONObject root = new JSONObject(json);
                java.util.Iterator<String> keys = root.keys();
                while (keys.hasNext()) {
                    String key = keys.next();
                    JSONObject o = root.getJSONObject(key);
                    map.put(key, new TypeInfo(key, o.getString("displayName"), readSeverity(o)));
                }
            } catch (Exception e) {
                Log.e(TAG, "Failed to load " + f, e);
            }
        }
        if (map.isEmpty()) {
            for (String type : ReferenceLibrary.FEATURE_TYPES) {
                map.put(type, new TypeInfo(type, ReferenceLibrary.displayName(type), Severity.CRITICAL));
            }
            persist(context, map);
        }
        return map;
    }

    /**
     * Reads the severity of one entry, accepting files written before the scale existed.
     *
     * A legacy {@code "critical": false} becomes {@link Severity#ADVISORY}, not
     * {@link Severity#IGNORED}. Those two were one value before, so there is no way to tell
     * which the operator meant, and promoting is the recoverable direction: an advisory type
     * still surfaces in the warning notes where it can be noticed and demoted, whereas an
     * auto-ignored one would vanish from every report with nothing to hint at why.
     */
    private static Severity readSeverity(JSONObject o) {
        if (o.has("severity")) return Severity.fromLevel(o.optInt("severity", Severity.CRITICAL.level));
        return o.optBoolean("critical", true) ? Severity.CRITICAL : Severity.ADVISORY;
    }

    private static void persist(Context context, Map<String, TypeInfo> map) {
        try {
            JSONObject root = new JSONObject();
            for (TypeInfo t : map.values()) {
                JSONObject o = new JSONObject();
                o.put("displayName", t.displayName);
                o.put("severity", t.severity.level);
                root.put(t.key, o);
            }
            try (FileOutputStream fos = new FileOutputStream(file(context))) {
                fos.write(root.toString().getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to persist feature type registry", e);
        }
    }

    private static File file(Context context) {
        return new File(context.getFilesDir(), FILE);
    }

}
