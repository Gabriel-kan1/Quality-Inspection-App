# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile

# ── Rules this app actually needs ─────────────────────────────────────────────

# Stack traces from the field are the only debugging signal an offline app gives back, so
# keep the line numbers; the file names are hidden so the trace does not double as a map of
# the source tree.
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# OpenCV's Java API is driven from native code: libopencv_java4 constructs Mat, MatOfPoint,
# Rect and the rest by class name through JNI, and the AAR ships no consumer rules. Renaming
# or stripping any of it crashes the first detection with NoSuchMethodError rather than at
# build time.
-keep class org.opencv.** { *; }
-dontwarn org.opencv.**

# ONNX Runtime's consumer rules only keep its telemetry classes. The inference path is
# the same story as OpenCV: libonnxruntime4j_jni resolves TensorInfo, OnnxTensor and the
# rest through GetMethodID by name, and the first OrtSession.run() aborts the process with
# "JNI DETECTED ERROR: java_class == null" if any of them was renamed. Found by running the
# minified build, not by reading the rules.
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Fragments are instantiated by the Navigation component from the class names in
# nav_graph.xml. aapt keeps classes named in layouts and the manifest, but not always those
# named in navigation resources, so pin them explicitly.
-keep class com.example.ui.fragment.** extends androidx.fragment.app.Fragment { <init>(); }
-keep class com.example.ui.view.** extends android.view.View { <init>(...); }

# Room entities and DAOs are covered by Room's consumer rules; Robolectric/Roborazzi are
# test-only and never enter a release build.
