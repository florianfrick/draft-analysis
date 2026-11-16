plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

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
        versionName = "0.1-mvp"

        // Where the Python backend lives. Local-only for now: no auth, no TLS.
        //
        // 10.0.2.2 is the Android emulator's alias for the *host's* loopback
        // interface, so frontend/serve.sh (which binds 127.0.0.1 only) is
        // reachable from the emulator with no server change at all.
        //
        // A physical device cannot see the host loopback. For that you must
        // start the server with --host 0.0.0.0 and put the machine's LAN
        // address here, plus that same address in
        // res/xml/network_security_config.xml.
        buildConfigField("String", "API_BASE", "\"http://10.0.2.2:8777\"")
    }

    buildFeatures {
        compose = true
        buildConfig = true
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
}
