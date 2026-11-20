plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

// Public Cloudflare Worker URL for the opt-in chat panel. This is a public
// endpoint, not a secret: the API key lives only in the Worker. Override with
//   gradle -PchatEndpoint=https://another-worker.workers.dev assembleDebug
val chatEndpoint = (findProperty("chatEndpoint") as String?)
    ?: "https://draft-assistant-chat.floriangfrick.workers.dev"

android {
    namespace = "com.draftassistant"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.draftassistant"
        minSdk = 24
        // Pinned to 34 on purpose. targetSdk 35 makes Android 15 force
        // edge-to-edge, which would draw the board under the status bar
        // unless the insets are handled explicitly. The MVP is not
        // edge-to-edge, so 34 is the honest value; bump it together with a
        // proper WindowInsets pass, not before.
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        buildConfigField("String", "CHAT_ENDPOINT", "\"$chatEndpoint\"")
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    // The two .pte weights are copied out of assets into filesDir at first
    // launch because Module.load needs a real path. Keep them uncompressed
    // so that copy is a straight read rather than a decompress.
    androidResources {
        noCompress += "pte"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)

    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)

    implementation(libs.okhttp)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.coil.compose)
    implementation(libs.executorch.android)

    testImplementation(libs.junit)
}
