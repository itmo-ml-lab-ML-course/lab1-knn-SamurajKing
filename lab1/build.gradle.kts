import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.9.22"
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

sourceSets {
    named("main") {
        java.srcDir("src/main/kotlin")
    }
}

dependencies {
    testImplementation(kotlin("test"))
    implementation("com.github.holgerbrandl:krangl:0.18.4")
    implementation("com.github.haifengl:smile-kotlin:3.0.3")
    implementation("org.jetbrains.kotlinx:kandy-lets-plot:0.6.0-dev-48")
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "21"
}