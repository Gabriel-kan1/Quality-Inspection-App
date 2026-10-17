package com.example.ml;

import android.graphics.RectF;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Runs the app's own edge-method selection and positional check over detections exported from
 * the 40 lighting test photos by the scratch export script (boxes and classifications per
 * method, per photo), so the selector and {@link FeatureLocalisation} are measured as shipped
 * rather than as a Python re-implementation.
 *
 * <p>Not part of the normal suite: it is skipped unless {@code TESTSET_STUDY_IN} names the
 * exported JSON. {@code TESTSET_STUDY_OUT} receives the results.
 *
 * <pre>
 *   TESTSET_STUDY_IN=export.json TESTSET_STUDY_OUT=out.json \
 *     ./gradlew :app:testDebugUnitTest --tests com.example.ml.TestPhotoStudy
 * </pre>
 */
@RunWith(RobolectricTestRunner.class)
public class TestPhotoStudy {

    private static final String[] ENCODERS = {"dino", "clip"};

    @Test
    public void run() throws Exception {
        String in = System.getenv("TESTSET_STUDY_IN");
        Assume.assumeTrue(in != null && new File(in).isFile());
        String outPath = System.getenv("TESTSET_STUDY_OUT");

        JSONObject root = new JSONObject(new String(Files.readAllBytes(new File(in).toPath()),
                StandardCharsets.UTF_8));
        JSONObject regions = root.getJSONObject("regions");
        JSONObject out = new JSONObject();

        for (java.util.Iterator<String> it = regions.keys(); it.hasNext(); ) {
            String regionName = it.next();
            JSONObject region = regions.getJSONObject(regionName);

            JSONArray features = region.getJSONObject("reference").getJSONArray("features");
            List<RectF> refRects = new ArrayList<>();
            List<String> refTypes = new ArrayList<>();
            Set<String> critical = new LinkedHashSet<>();
            Map<String, Integer> expected = new HashMap<>();
            for (int i = 0; i < features.length(); i++) {
                JSONObject f = features.getJSONObject(i);
                refRects.add(rect(f.getJSONArray("rect")));
                refTypes.add(f.getString("type"));
                critical.add(f.getString("type"));
                expected.merge(f.getString("type"), 1, Integer::sum);
            }
            FeatureLayout layout = FeatureLocalisation.layoutFrom(refRects, refTypes, critical);
            // The reference's own method wins a tie, as ReferenceFit.referenceMethodId does.
            String incumbent = region.getJSONObject("reference").optString("method", "v6");
            ReferenceFit base = ReferenceFit.from(layout);

            JSONObject regionOut = new JSONObject();
            JSONObject photos = region.getJSONObject("photos");
            for (java.util.Iterator<String> pit = photos.keys(); pit.hasNext(); ) {
                String variant = pit.next();
                JSONObject photo = photos.getJSONObject(variant);
                JSONObject methods = photo.getJSONObject("methods");
                JSONObject photoOut = new JSONObject();
                photoOut.put("kind", photo.getString("kind"));

                for (String encoder : ENCODERS) {
                    Map<String, List<RectF>> rects = new LinkedHashMap<>();
                    Map<String, List<String>> types = new LinkedHashMap<>();
                    Map<String, Double> scores = new LinkedHashMap<>();
                    Map<String, ReferenceFit.Fit> fits = new HashMap<>();
                    Map<String, Integer> agreeing = new HashMap<>();

                    for (java.util.Iterator<String> mit = methods.keys(); mit.hasNext(); ) {
                        String m = mit.next();
                        JSONArray dets = methods.getJSONArray(m);
                        List<RectF> r = new ArrayList<>();
                        List<String> t = new ArrayList<>();
                        for (int i = 0; i < dets.length(); i++) {
                            JSONObject d = dets.getJSONObject(i);
                            r.add(rect(d.getJSONArray("rect")));
                            t.add(d.getJSONObject(encoder).getString("type"));
                        }
                        rects.put(m, r);
                        types.put(m, t);
                        ReferenceFit.Fit fit = base.fit(r);
                        fits.put(m, fit);
                        scores.put(m, fit.score);
                        Map<RectF, String> lookup = new IdentityHashMap<>();
                        for (int i = 0; i < r.size(); i++) lookup.put(r.get(i), t.get(i));
                        agreeing.put(m, base.withLabeller(lookup::get).labelsAgreeing(r, fit));
                    }

                    // The adaptive stage: position decides, labels break close fits, the
                    // reference's own method wins a tie.
                    List<String> shortlist = AdaptivePreprocessing.shortlist(scores,
                            AdaptivePreprocessing.SHORTLIST_MARGIN,
                            AdaptivePreprocessing.MAX_LABELLED);
                    Map<String, Integer> labels = new HashMap<>();
                    for (String m : shortlist) labels.put(m, agreeing.get(m));
                    String chosen = AdaptivePreprocessing.choose(incumbent, scores, labels,
                            AdaptivePreprocessing.SHORTLIST_MARGIN);

                    JSONObject encOut = new JSONObject();
                    encOut.put("chosen", chosen);
                    encOut.put("shortlist", new JSONArray(shortlist));
                    JSONObject perMethod = new JSONObject();
                    for (String m : rects.keySet()) {
                        perMethod.put(m, verdict(layout, critical, expected, rects.get(m),
                                types.get(m), fits.get(m), agreeing.get(m)));
                    }
                    encOut.put("methods", perMethod);
                    photoOut.put(encoder, encOut);
                }
                regionOut.put(variant, photoOut);
            }
            out.put(regionName, regionOut);
        }

        if (outPath != null) {
            Files.write(new File(outPath).toPath(), out.toString(1).getBytes(StandardCharsets.UTF_8));
        }
    }

    /** The results screen's verdict: per-type critical counts, then the positional check. */
    private static JSONObject verdict(FeatureLayout layout, Set<String> critical,
                                      Map<String, Integer> expected, List<RectF> rects,
                                      List<String> types, ReferenceFit.Fit fit, int agreeing)
            throws Exception {
        Map<String, Integer> detected = new HashMap<>();
        List<RectF> locRects = new ArrayList<>();
        List<String> locTypes = new ArrayList<>();
        for (int i = 0; i < types.size(); i++) {
            String t = types.get(i);
            if (critical.contains(t)) detected.merge(t, 1, Integer::sum);
            if (critical.contains(t) || ReferenceLibrary.UNKNOWN.equals(t)) {
                locRects.add(rects.get(i));
                locTypes.add(t);
            }
        }
        boolean countFail = false;
        for (String t : critical) {
            if (!expected.getOrDefault(t, 0).equals(detected.getOrDefault(t, 0))) countFail = true;
        }
        FeatureLocalisation.Result loc = FeatureLocalisation.localise(layout, locRects, locTypes,
                critical);
        boolean positional = loc.aligned && loc.hasPositionalDefect();

        JSONObject o = new JSONObject();
        o.put("boxes", rects.size());
        o.put("fitScore", fit.score);
        o.put("matched", fit.matched);
        o.put("extras", fit.extras);
        o.put("labelsAgreeing", agreeing);
        o.put("boxForFeature", new JSONArray(fit.boxForFeature));
        o.put("countVerdict", countFail ? "FAIL" : "PASS");
        o.put("aligned", loc.aligned);
        o.put("positionalDefect", positional);
        o.put("verdict", countFail || positional ? "FAIL" : "PASS");
        JSONObject statuses = new JSONObject();
        if (loc.aligned) {
            for (FeatureLocalisation.Status s : FeatureLocalisation.Status.values()) {
                statuses.put(s.name(), loc.countWithStatus(s));
            }
        } else {
            o.put("unavailable", loc.unavailableReason);
        }
        o.put("slots", statuses);
        return o;
    }

    private static RectF rect(JSONArray a) throws Exception {
        return new RectF((float) a.getDouble(0), (float) a.getDouble(1),
                (float) a.getDouble(2), (float) a.getDouble(3));
    }
}
