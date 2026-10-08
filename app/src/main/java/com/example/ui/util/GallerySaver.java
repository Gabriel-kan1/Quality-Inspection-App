package com.example.ui.util;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Toast;

import com.example.ml.HardwareFeatureExtractor;

import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Saves a photo into the phone's gallery: the "Chassis QC" album under Pictures, where the
 * gallery app shows it and Google Photos backs it up when the operator has device-folder backup
 * turned on.
 *
 * <p>Through MediaStore, which on Android 10 and later needs no storage permission: the app may
 * add to shared media without being able to read anyone else's. Older versions would need the
 * storage permission, which this app does not ask for, so there the action says so instead.
 */
public final class GallerySaver {

    private static final String TAG = "GallerySaver";
    /** The album, under Pictures. */
    public static final String ALBUM = "Chassis QC";

    private GallerySaver() {}

    /** Whether this Android version can save without a storage permission. */
    public static boolean supported() {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q;
    }

    /** A name that says what the photo is and when, e.g. {@code REFERENCE_PART-A03_20261008_141502}. */
    public static String name(String kind, String partId) {
        String part = partId == null ? "" : partId.replaceAll("[^A-Za-z0-9_-]", "_");
        return kind + "_" + part + "_"
                + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
    }

    /**
     * Copies a photo's original file — full resolution, EXIF intact — into the gallery, on a
     * background thread, and says how it went.
     *
     * @param sourceUri any URI the pipeline takes: content, file or bundled asset
     */
    public static void savePhoto(Context context, String sourceUri, String displayName) {
        final Context app = context.getApplicationContext();
        if (!supported()) {
            tellUnsupported(app);
            return;
        }
        final boolean png = sourceUri != null && sourceUri.toLowerCase(Locale.US).endsWith(".png");
        new Thread(() -> {
            Uri saved = null;
            try (InputStream in = HardwareFeatureExtractor.open(app, sourceUri)) {
                if (in == null) throw new IllegalStateException("cannot read " + sourceUri);
                saved = write(app, displayName, png, out -> {
                    byte[] buffer = new byte[64 * 1024];
                    int n;
                    while ((n = in.read(buffer)) != -1) out.write(buffer, 0, n);
                });
            } catch (Exception e) {
                Log.e(TAG, "Saving " + sourceUri + " to the gallery failed", e);
            }
            report(app, saved != null);
        }).start();
    }

    /** Saves an image made in the app — an annotated overlay, say — as a JPEG. Recycles it. */
    public static void saveBitmap(Context context, Bitmap bitmap, String displayName) {
        final Context app = context.getApplicationContext();
        if (!supported()) {
            tellUnsupported(app);
            bitmap.recycle();
            return;
        }
        new Thread(() -> {
            Uri saved = null;
            try {
                saved = write(app, displayName, false,
                        out -> bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out));
            } catch (Exception e) {
                Log.e(TAG, "Saving an image to the gallery failed", e);
            } finally {
                bitmap.recycle();
            }
            report(app, saved != null);
        }).start();
    }

    private interface Writer {
        void writeTo(OutputStream out) throws Exception;
    }

    /**
     * One gallery entry, written while marked pending so the gallery never shows a half-written
     * file, and removed again if writing fails.
     */
    private static Uri write(Context app, String displayName, boolean png, Writer writer)
            throws Exception {
        ContentResolver resolver = app.getContentResolver();
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, displayName + (png ? ".png" : ".jpg"));
        values.put(MediaStore.Images.Media.MIME_TYPE, png ? "image/png" : "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH,
                Environment.DIRECTORY_PICTURES + "/" + ALBUM);
        values.put(MediaStore.Images.Media.IS_PENDING, 1);
        Uri uri = resolver.insert(
                MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values);
        if (uri == null) throw new IllegalStateException("the gallery refused a new entry");
        try (OutputStream out = resolver.openOutputStream(uri)) {
            if (out == null) throw new IllegalStateException("cannot write the new entry");
            writer.writeTo(out);
        } catch (Exception e) {
            resolver.delete(uri, null, null);
            throw e;
        }
        values.clear();
        values.put(MediaStore.Images.Media.IS_PENDING, 0);
        resolver.update(uri, values, null, null);
        return uri;
    }

    private static void report(Context app, boolean ok) {
        new Handler(Looper.getMainLooper()).post(() -> Toast.makeText(app, ok
                ? "Saved to the gallery, in Pictures/" + ALBUM + "."
                : "Could not save to the gallery.", Toast.LENGTH_SHORT).show());
    }

    private static void tellUnsupported(Context app) {
        Toast.makeText(app, "Saving to the gallery needs Android 10 or later.",
                Toast.LENGTH_LONG).show();
    }
}
