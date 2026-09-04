plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.compose)
  alias(libs.plugins.google.devtools.ksp)
  alias(libs.plugins.roborazzi)
}

android {
  namespace = "com.example"
  compileSdk = 35

  defaultConfig {
    applicationId = "com.aistudio.qualityinspection.bzpqrk"
    minSdk = 24
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

    // OpenCV + ONNX Runtime ship native libs for four ABIs, which pushed the debug
    // APK past 350MB and made every install-and-run cycle painfully slow. x86_64
    // covers emulators, arm64-v8a covers modern physical devices; the dropped 32-bit
    // ABIs (x86, armeabi-v7a) have no supported target left worth the ~180MB.
    ndk { abiFilters += listOf("x86_64", "arm64-v8a") }
  }

  signingConfigs {
    create("release") {
      val keystorePath = System.getenv("KEYSTORE_PATH") ?: "${rootDir}/my-upload-key.jks"
      storeFile = file(keystorePath)
      storePassword = System.getenv("STORE_PASSWORD")
      keyAlias = "upload"
      keyPassword = System.getenv("KEY_PASSWORD")
    }
  }

  buildTypes {
    release {
      isCrunchPngs = false
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
      signingConfig = signingConfigs.getByName("release")
    }
    debug {
      // Uses the default debug signing config
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
  }
  buildFeatures {
    compose = true
    viewBinding = true
    buildConfig = true
  }
  // Keep the CLIP ONNX model uncompressed so ONNX Runtime can memory-map it
  // directly from the APK instead of decompressing a ~45MB asset into RAM.
  androidResources { noCompress += "onnx" }
  testOptions { unitTests { isIncludeAndroidResources = true } }
}

// Some unused dependencies are commented out below instead of being removed.
// This makes it easy to add them back in the future if needed.
//
// Retrofit, OkHttp, Moshi, the Firebase BOM and ML Kit barcode scanning were removed outright
// rather than commented out. None of them had a single call site, and this app holds no INTERNET
// permission on purpose — shipping unused networking code in an offline app is attack surface and
// install size for nothing. The Gemini scaffolding that came with the AI Studio template (the
// secrets plugin, .env.example, metadata.json) went with them, so there is no longer a slot for an
// API key to be dropped into by accident.
dependencies {
  implementation(platform(libs.androidx.compose.bom))
  // implementation(libs.accompanist.permissions)
  implementation(libs.androidx.activity.compose)
  implementation(libs.androidx.camera.camera2)
  implementation(libs.androidx.camera.core)
  implementation(libs.androidx.camera.lifecycle)
  implementation(libs.androidx.camera.view)
  implementation(libs.androidx.navigation.fragment)
  implementation(libs.androidx.navigation.ui)
  implementation(libs.google.material)
  implementation(libs.androidx.compose.material.icons.core)
  // implementation(libs.androidx.compose.material.icons.extended)
  implementation(libs.androidx.compose.material3)
  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.ui.graphics)
  implementation(libs.androidx.compose.ui.tooling.preview)
  implementation(libs.androidx.core.ktx)
  // implementation(libs.androidx.datastore.preferences)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.lifecycle.viewmodel.compose)
  // implementation(libs.androidx.navigation.compose)
  // implementation(libs.coil.compose)
  implementation(libs.kotlinx.coroutines.android)
  implementation(libs.kotlinx.coroutines.core)
  // implementation(libs.play.services.location)
  implementation(libs.opencv)
  implementation(libs.onnxruntime.android)
  // Camera JPEGs and gallery picks carry orientation in EXIF rather than in the pixels;
  // the encoder is not rotation invariant, so every disk load has to apply it.
  implementation(libs.androidx.exifinterface)
  // QR generation for the traceability code on each report. ZXing core is pure Java,
  // Apache-2.0, and encodes as well as decodes — ML Kit only decodes, and even its bundled
  // variant would pull Play Services back into an app that deliberately removed it above.
  implementation(libs.zxing.core)
  testImplementation(libs.androidx.compose.ui.test.junit4)
  testImplementation(libs.androidx.core)
  testImplementation(libs.androidx.junit)
  testImplementation(libs.junit)
  testImplementation(libs.kotlinx.coroutines.test)
  testImplementation(libs.robolectric)
  testImplementation(libs.roborazzi)
  testImplementation(libs.roborazzi.compose)
  testImplementation(libs.roborazzi.junit.rule)
  androidTestImplementation(platform(libs.androidx.compose.bom))
  androidTestImplementation(libs.androidx.compose.ui.test.junit4)
  androidTestImplementation(libs.androidx.espresso.core)
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.runner)
  debugImplementation(libs.androidx.compose.ui.test.manifest)
  debugImplementation(libs.androidx.compose.ui.tooling)

  // Room (Java-based implementation requires annotationProcessor)
  implementation(libs.androidx.room.runtime)
  implementation(libs.androidx.room.ktx)
  annotationProcessor(libs.androidx.room.compiler)
  ksp(libs.androidx.room.compiler)
}
