package com.example.ui;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import androidx.navigation.NavAction;
import androidx.navigation.NavController;
import androidx.navigation.NavDestination;
import androidx.navigation.NavOptions;
import androidx.navigation.Navigation;
import androidx.test.core.app.ActivityScenario;

import com.example.MainActivity;
import com.example.R;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/**
 * Filing a report must take the results screen off the back stack.
 *
 * <p>Left there, BACK from the report re-created it: the fragment re-ran its analysis, put up
 * a fresh spot-check prompt, and offered SAVE REPORT again — which filed a second, sealed
 * record for the same inspection. Seen on a device as QC-RECORD-1 and QC-RECORD-2 for one
 * photograph, one minute apart. The fix is one attribute in the navigation graph, and it is
 * exactly the kind of attribute that gets lost in a merge, so it is pinned here.
 */
@RunWith(RobolectricTestRunner.class)
public class FiledReportBackStackTest {

    @Test
    public void filingAReportPopsTheResultsScreen() {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            scenario.onActivity(activity -> {
                NavController nav = Navigation.findNavController(activity, R.id.nav_host_fragment);
                NavDestination results = nav.getGraph().findNode(R.id.navigation_results);
                assertNotNull("results screen is in the graph", results);

                NavAction toReport = results.getAction(R.id.action_results_to_report);
                assertNotNull("results -> report action exists", toReport);
                assertEquals(R.id.navigation_report, toReport.getDestinationId());

                NavOptions options = toReport.getNavOptions();
                assertNotNull("the action must carry navigation options", options);
                assertEquals("filing pops back to the results destination itself",
                        R.id.navigation_results, options.getPopUpToId());
                assertTrue("and removes it too, so BACK cannot land on a stale results screen",
                        options.isPopUpToInclusive());
            });
        }
    }

    /** The scanner already does this; the two must not drift apart. */
    @Test
    public void scanningAReportPopsTheScannerTheSameWay() {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            scenario.onActivity(activity -> {
                NavController nav = Navigation.findNavController(activity, R.id.nav_host_fragment);
                NavAction toReport = nav.getGraph().findNode(R.id.navigation_report_scan)
                        .getAction(R.id.action_scan_to_report);
                assertNotNull(toReport);
                assertEquals(R.id.navigation_report_scan, toReport.getNavOptions().getPopUpToId());
                assertTrue(toReport.getNavOptions().isPopUpToInclusive());
            });
        }
    }
}
