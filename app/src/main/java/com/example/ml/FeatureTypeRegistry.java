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
 * User-editable catalog of feature types: display name + whether the type is
 * "critical" (should appear in the exported report) or not (e.g. cosmetic
 * defects like scratch/rust that an operator wants tracked but not reported).
 *
 * Seeded once from {@link ReferenceLibrary#FEATURE_TYPES} (all critical by
 * default), then persisted to filesDir/feature_types.json. New types are added
 * on-device when the user labels an "unknown" detection during inspection —
 * criticality is always an explicit user choice, never inferred from the name.
 */
public final class FeatureTypeRegistry {

    private static final String TAG = "FeatureTypeRegistry";
    private static final String FILE = "feature_types.json";

    public static final class TypeInfo {
        public final String key;
        public final String displayName;
        public final boolean critical;

        public TypeInfo(String key, String displayName, boolean critical) {
            this.key = key;
            this.displayName = displayName;
            this.critical = critical;
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

    /** Defaults to critical=true for anything not explicitly registered, so nothing silently drops out of a report. */
    public static synchronized boolean isCritical(Context context, String key) {
        TypeInfo t = get(context, key);
        return t == null || t.critical;
    }

    public static synchronized String displayName(Context context, String key) {
        TypeInfo t = get(context, key);
        if (t != null) return t.displayName;
        return ReferenceLibrary.displayName(key);
    }

    /** Adds a new type or updates an existing one's display name/criticality. */
    public static synchronized void upsert(Context context, String key, String displayName, boolean critical) {
        if (key == null || key.isEmpty()) return;
        Map<String, TypeInfo> all = load(context);
        all.put(key, new TypeInfo(key, displayName, critical));
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
                String json = readAll(is);
                JSONObject root = new JSONObject(json);
                java.util.Iterator<String> keys = root.keys();
                while (keys.hasNext()) {
                    String key = keys.next();
                    JSONObject o = root.getJSONObject(key);
                    map.put(key, new TypeInfo(key, o.getString("displayName"), o.getBoolean("critical")));
                }
            } catch (Exception e) {
                Log.e(TAG, "Failed to load " + f, e);
            }
        }
        if (map.isEmpty()) {
            for (String type : ReferenceLibrary.FEATURE_TYPES) {
                map.put(type, new TypeInfo(type, ReferenceLibrary.displayName(type), true));
            }
            persist(context, map);
        }
        return map;
    }

    private static void persist(Context context, Map<String, TypeInfo> map) {
        try {
            JSONObject root = new JSONObject();
            for (TypeInfo t : map.values()) {
                JSONObject o = new JSONObject();
                o.put("displayName", t.displayName);
                o.put("critical", t.critical);
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

    private static String readAll(InputStream is) throws Exception {
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        int n;
        while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
        return new String(bos.toByteArray(), StandardCharsets.UTF_8);
    }
}
