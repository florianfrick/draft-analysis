# Keep kotlinx.serialization metadata for the @Serializable DTOs.
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.**

-keepclassmembers class com.draftassistant.data.** {
    *** Companion;
}
-keepclasseswithmembers class com.draftassistant.data.** {
    kotlinx.serialization.KSerializer serializer(...);
}
