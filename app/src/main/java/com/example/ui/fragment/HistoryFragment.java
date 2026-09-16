package com.example.ui.fragment;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.LruCache;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.InspectionRecord;
import com.example.databinding.FragmentHistoryBinding;
import com.example.ml.HardwareFeatureExtractor;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class HistoryFragment extends Fragment {

    private FragmentHistoryBinding binding;
    private List<InspectionRecord> inspectionRecords = new ArrayList<>();
    private final List<InspectionRecord> filteredRecords = new ArrayList<>();
    private HistoryAdapter adapter;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentHistoryBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        binding.recyclerHistoryLogs.setLayoutManager(new LinearLayoutManager(getContext()));
        adapter = new HistoryAdapter();
        binding.recyclerHistoryLogs.setAdapter(adapter);

        binding.btnScanReport.setOnClickListener(v ->
                Navigation.findNavController(v).navigate(R.id.action_history_to_scan));

        binding.etSearch.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) { }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                filterCollection(s.toString());
            }

            @Override
            public void afterTextChanged(Editable s) { }
        });

        loadHistoryFromDatabase();
    }

    private void loadHistoryFromDatabase() {
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
            List<InspectionRecord> list = db.inspectionDao().getAllInspections();

            requireActivity().runOnUiThread(() -> {
                if (list != null) {
                    inspectionRecords = list;
                    filterCollection(binding.etSearch.getText().toString());
                }
            });
        }).start();
    }

    private void filterCollection(String query) {
        filteredRecords.clear();
        if (query == null || query.trim().isEmpty()) {
            filteredRecords.addAll(inspectionRecords);
        } else {
            String lowercaseQuery = query.toLowerCase(Locale.getDefault()).trim();
            for (InspectionRecord rec : inspectionRecords) {
                if (rec.getPartId() != null && rec.getPartId().toLowerCase(Locale.getDefault()).contains(lowercaseQuery)) {
                    filteredRecords.add(rec);
                }
            }
        }

        if (filteredRecords.isEmpty()) {
            binding.tvEmptyHistory.setVisibility(View.VISIBLE);
            binding.recyclerHistoryLogs.setVisibility(View.GONE);
        } else {
            binding.tvEmptyHistory.setVisibility(View.GONE);
            binding.recyclerHistoryLogs.setVisibility(View.VISIBLE);
        }
        
        adapter.notifyDataSetChanged();
    }

    /** Longest edge a row thumbnail is decoded at. The view is ~56dp; this is plenty. */
    private static final int THUMB_PX = 160;

    /** Decoded thumbnails by image URI, so scrolling back over a row does not decode again. */
    private final LruCache<String, Bitmap> thumbs = new LruCache<>(64);
    private final ExecutorService thumbExecutor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    /**
     * Puts a record's photograph on its row without decoding on the main thread.
     *
     * <p>This replaces {@code ImageView.setImageURI}, which was wrong three ways at once: it
     * decodes synchronously on whatever thread calls it, so every row bind stalled the list
     * for a full camera-resolution decode; it decodes at full size, tens of megabytes for a
     * 56dp view; and it cannot read {@code file:///android_asset/} at all, so any inspection
     * made from a bundled photograph showed a blank and logged ENOENT. loadBitmap already
     * handles assets, sampling and EXIF for the rest of the app.
     *
     * <p>Recycled rows are the race: a decode started for one record can finish after the
     * view has been rebound to another. The view carries the URI it currently wants as a
     * tag, and a decode only lands if that tag still matches.
     */
    private void bindThumbnail(ImageView view, @Nullable String uri) {
        if (uri == null || uri.isEmpty() || "mock_image_uri".equals(uri)) {
            view.setTag(null);
            view.setImageResource(android.R.drawable.ic_menu_camera);
            return;
        }
        view.setTag(uri);
        Bitmap cached = thumbs.get(uri);
        if (cached != null) {
            view.setImageBitmap(cached);
            return;
        }
        view.setImageResource(android.R.drawable.ic_menu_camera);
        if (thumbExecutor.isShutdown()) return;
        final android.content.Context context = requireContext().getApplicationContext();
        thumbExecutor.execute(() -> {
            Bitmap bitmap = HardwareFeatureExtractor.loadBitmap(context, uri, THUMB_PX);
            if (bitmap == null) return;
            mainHandler.post(() -> {
                if (binding == null) {
                    bitmap.recycle();
                    return;
                }
                thumbs.put(uri, bitmap);
                if (uri.equals(view.getTag())) view.setImageBitmap(bitmap);
            });
        });
    }

    // Inner class Adapter representing standard clean Java Recycler adapter structure 
    private class HistoryAdapter extends RecyclerView.Adapter<HistoryAdapter.ViewHolder> {

        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_history_row, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            InspectionRecord rec = filteredRecords.get(position);

            holder.tvPartId.setText(rec.getPartId());
            
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault());
            String dateFormatted = sdf.format(new Date(rec.getTimestamp()));
            holder.tvTime.setText(dateFormatted);

            bindThumbnail(holder.ivThumb, rec.getImageUri());

            // Three states, not two: an inspection run without a reference image compared
            // nothing, and must not fall through to FAIL any more than to PASS.
            if ("PASS".equalsIgnoreCase(rec.getPassOrFail())) {
                holder.tvStatus.setText("PASS");
                holder.tvStatus.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass));
            } else if ("UNVERIFIED".equalsIgnoreCase(rec.getPassOrFail())) {
                holder.tvStatus.setText("UNVERIFIED");
                holder.tvStatus.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.orange_warn));
            } else {
                holder.tvStatus.setText("FAIL");
                holder.tvStatus.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
            }

            holder.itemView.setOnClickListener(v -> {
                Bundle args = new Bundle();
                args.putLong("inspectionId", rec.getId());
                Navigation.findNavController(v).navigate(R.id.action_history_to_report, args);
            });
        }

        @Override
        public int getItemCount() {
            return filteredRecords.size();
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            final ImageView ivThumb;
            final TextView tvPartId;
            final TextView tvTime;
            final TextView tvStatus;

            ViewHolder(View itemView) {
                super(itemView);
                ivThumb = itemView.findViewById(R.id.iv_record_thumb);
                tvPartId = itemView.findViewById(R.id.tv_record_part_id);
                tvTime = itemView.findViewById(R.id.tv_record_time);
                tvStatus = itemView.findViewById(R.id.tv_record_status_badge);
            }
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        thumbExecutor.shutdownNow();
        mainHandler.removeCallbacksAndMessages(null);
        thumbs.evictAll();
        binding = null;
    }
}
