package com.example;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.MotionEvent;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;
import androidx.navigation.ui.NavigationUI;

import com.example.R;
import com.example.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    /** How long the status and navigation bars stay up after the operator summons them. */
    private static final long SYSTEM_BAR_HIDE_DELAY_MS = 3000L;

    private ActivityMainBinding binding;
    private NavController navController;

    private final Handler uiHandler = new Handler(Looper.getMainLooper());
    private final Runnable hideSystemBars = this::hideSystemBars;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // The phone's own bars steal height from a screen that is mostly camera preview and
        // overlay, so they are dropped a few seconds after every appearance. Swiping from an
        // edge brings them back transiently — they overlay rather than resize the layout —
        // and the timer below sends them away again.
        WindowInsetsControllerCompat controller = getInsetsController();
        controller.setSystemBarsBehavior(
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
        ViewCompat.setOnApplyWindowInsetsListener(binding.getRoot(), (v, insets) -> {
            if (insets.isVisible(WindowInsetsCompat.Type.systemBars())) scheduleHideSystemBars();
            return insets;
        });
        scheduleHideSystemBars();

        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager()
                .findFragmentById(R.id.nav_host_fragment);
                
        if (navHostFragment != null) {
            navController = navHostFragment.getNavController();
            NavigationUI.setupWithNavController(binding.bottomNavigation, navController);
            
            // Routes the global back button through the same dispatcher as the system
            // back gesture, so both behave identically on every screen.
            binding.btnGlobalBack.setOnClickListener(v -> getOnBackPressedDispatcher().onBackPressed());

            // Show bottom navigation only on Top-Level screens (Home & History)
            navController.addOnDestinationChangedListener((controller2, destination, arguments) -> {
                int id = destination.getId();
                if (id == R.id.navigation_home || id == R.id.navigation_history) {
                    binding.bottomNavigation.setVisibility(View.VISIBLE);
                } else {
                    binding.bottomNavigation.setVisibility(View.GONE);
                }

                // Home is the start destination: going back from there would exit the app.
                binding.layoutBackBar.setVisibility(
                        id == R.id.navigation_home ? View.GONE : View.VISIBLE);
            });
        }
    }

    private WindowInsetsControllerCompat getInsetsController() {
        return androidx.core.view.WindowCompat.getInsetsController(
                getWindow(), getWindow().getDecorView());
    }

    /** Restarts the countdown, so a fresh appearance always gets its full three seconds. */
    private void scheduleHideSystemBars() {
        uiHandler.removeCallbacks(hideSystemBars);
        uiHandler.postDelayed(hideSystemBars, SYSTEM_BAR_HIDE_DELAY_MS);
    }

    private void hideSystemBars() {
        getInsetsController().hide(WindowInsetsCompat.Type.systemBars());
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent event) {
        // A swipe that pulls the bars back in usually lands here first; re-arming on any touch
        // also stops the bars vanishing mid-gesture while the operator is reaching for them.
        if (event.getActionMasked() == MotionEvent.ACTION_DOWN) scheduleHideSystemBars();
        return super.dispatchTouchEvent(event);
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) scheduleHideSystemBars();
        else uiHandler.removeCallbacks(hideSystemBars);
    }

    @Override
    protected void onDestroy() {
        uiHandler.removeCallbacks(hideSystemBars);
        super.onDestroy();
    }

    @Override
    public boolean onSupportNavigateUp() {
        if (navController != null) {
            return navController.navigateUp() || super.onSupportNavigateUp();
        }
        return super.onSupportNavigateUp();
    }
}
