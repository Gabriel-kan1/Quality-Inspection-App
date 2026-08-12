package com.example;

import android.os.Bundle;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;
import androidx.navigation.ui.NavigationUI;

import com.example.R;
import com.example.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private NavController navController;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager()
                .findFragmentById(R.id.nav_host_fragment);
                
        if (navHostFragment != null) {
            navController = navHostFragment.getNavController();
            NavigationUI.setupWithNavController(binding.bottomNavigation, navController);
            
            // Routes the global back button through the same dispatcher as the system
            // back gesture, so both behave identically on every screen.
            binding.btnGlobalBack.setOnClickListener(v -> getOnBackPressedDispatcher().onBackPressed());

            // Show bottom navigation only on Top-Level screens (Home & History)
            navController.addOnDestinationChangedListener((controller, destination, arguments) -> {
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

    @Override
    public boolean onSupportNavigateUp() {
        if (navController != null) {
            return navController.navigateUp() || super.onSupportNavigateUp();
        }
        return super.onSupportNavigateUp();
    }
}
