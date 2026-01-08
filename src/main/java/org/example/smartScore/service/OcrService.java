package org.example.smartScore.service;

import com.google.cloud.vision.v1.*;
import com.google.protobuf.ByteString;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
@org.springframework.boot.autoconfigure.condition.ConditionalOnBean(com.google.cloud.vision.v1.ImageAnnotatorClient.class)
public class OcrService {

    private final ImageAnnotatorClient imageAnnotatorClient;

    public String extractTextFromImage(byte[] imageBytes) throws IOException {
        try {
            ByteString imgBytes = ByteString.copyFrom(imageBytes);
            
            Image img = Image.newBuilder().setContent(imgBytes).build();
            Feature feat = Feature.newBuilder().setType(Feature.Type.TEXT_DETECTION).build();
            AnnotateImageRequest request =
                    AnnotateImageRequest.newBuilder()
                            .addFeatures(feat)
                            .setImage(img)
                            .build();
            
            BatchAnnotateImagesResponse response = imageAnnotatorClient.batchAnnotateImages(List.of(request));
            List<AnnotateImageResponse> responses = response.getResponsesList();
            
            AnnotateImageResponse res = responses.get(0);
            if (res.hasError()) {
                log.error("Error: {}", res.getError().getMessage());
                throw new IOException("OCR API Error: " + res.getError().getMessage());
            }
            
            // Extract full text annotation
            String fullText = "";
            if (res.hasFullTextAnnotation()) {
                fullText = res.getFullTextAnnotation().getText();
            } else {
                // Fallback to individual text annotations
                StringBuilder textBuilder = new StringBuilder();
                for (EntityAnnotation annotation : res.getTextAnnotationsList()) {
                    if (!annotation.getDescription().isEmpty()) {
                        textBuilder.append(annotation.getDescription()).append("\n");
                    }
                }
                fullText = textBuilder.toString();
            }
            
            log.debug("Extracted text length: {}", fullText.length());
            return fullText.trim();
        } catch (Exception e) {
            log.error("Error during OCR processing", e);
            throw new IOException("OCR processing failed", e);
        }
    }

    public List<String> extractAnswersFromImage(byte[] imageBytes) throws IOException {
        String fullText = extractTextFromImage(imageBytes);
        
        // 텍스트를 줄 단위로 분리하고, 숫자 답안만 추출
        String[] lines = fullText.split("\n");
        List<String> answers = new ArrayList<>();
        
        for (String line : lines) {
            String trimmed = line.trim();
            // 숫자만 포함된 라인을 답안으로 간주
            if (trimmed.matches("\\d+")) {
                answers.add(trimmed);
            } else if (trimmed.matches(".*\\d+.*")) {
                // 숫자가 포함된 경우 숫자만 추출
                String numbers = trimmed.replaceAll("[^0-9]", "");
                if (!numbers.isEmpty()) {
                    answers.add(numbers);
                }
            }
        }
        
        log.info("Extracted {} answers from image", answers.size());
        return answers;
    }
}

