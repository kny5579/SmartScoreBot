package org.example.smartScore.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.vision.v1.ImageAnnotatorClient;
import com.google.cloud.vision.v1.ImageAnnotatorSettings;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.FileInputStream;
import java.io.IOException;

@Slf4j
@Configuration
public class GoogleCloudConfig {

    @Value("${google.cloud.vision.credentials.path:}")
    private String credentialsPath;

    @Value("${google.cloud.vision.enabled:false}")
    private boolean visionEnabled;

    @Bean
    @ConditionalOnProperty(name = "google.cloud.vision.enabled", havingValue = "true", matchIfMissing = false)
    public ImageAnnotatorClient imageAnnotatorClient() throws IOException {
        ImageAnnotatorSettings.Builder settingsBuilder = ImageAnnotatorSettings.newBuilder();
        
        // 환경변수나 설정 파일에서 credentials 경로가 제공된 경우
        if (credentialsPath != null && !credentialsPath.isEmpty()) {
            try (FileInputStream credentialsStream = new FileInputStream(credentialsPath)) {
                GoogleCredentials credentials = GoogleCredentials.fromStream(credentialsStream);
                settingsBuilder.setCredentialsProvider(() -> credentials);
                log.info("Google Cloud credentials loaded from: {}", credentialsPath);
            }
        } else if (System.getenv("GOOGLE_APPLICATION_CREDENTIALS") != null) {
            // 환경변수 GOOGLE_APPLICATION_CREDENTIALS 사용
            String envPath = System.getenv("GOOGLE_APPLICATION_CREDENTIALS");
            try (FileInputStream credentialsStream = new FileInputStream(envPath)) {
                GoogleCredentials credentials = GoogleCredentials.fromStream(credentialsStream);
                settingsBuilder.setCredentialsProvider(() -> credentials);
                log.info("Google Cloud credentials loaded from environment variable: {}", envPath);
            }
        } else {
            // Application Default Credentials (ADC) 사용
            log.info("Using Application Default Credentials for Google Cloud Vision API");
        }
        
        return ImageAnnotatorClient.create(settingsBuilder.build());
    }
}

