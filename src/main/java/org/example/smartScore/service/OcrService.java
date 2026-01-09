package org.example.smartScore.service;

import com.google.cloud.vision.v1.*;
import com.google.protobuf.ByteString;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

    /**
     * 이미지에서 답안을 추출합니다.
     * 문제 번호와 답안이 함께 있는 경우 (예: "1: 3", "1번: 3", "1. 3") 문제 번호를 기준으로 매핑합니다.
     * 문제 번호가 없는 경우 순서대로 추출합니다.
     * 
     * @param imageBytes 이미지 바이트 배열
     * @return 문제 번호를 키로 하는 답안 맵 (문제 번호가 없는 경우 null 키 사용)
     */
    public Map<Integer, String> extractAnswersWithQuestionNumbers(byte[] imageBytes) throws IOException {
        String fullText = extractTextFromImage(imageBytes);
        
        // 텍스트를 줄 단위로 분리
        String[] lines = fullText.split("\n");
        Map<Integer, String> answerMap = new LinkedHashMap<>();
        int sequentialIndex = 1; // 문제 번호가 없는 경우를 위한 순차 인덱스
        
        // 문제 번호 패턴: "1:", "1번:", "1.", "1번", "1 " 등
        Pattern questionNumberPattern = Pattern.compile("(\\d+)[번:.]?\\s*([0-9]+)");
        
        for (String line : lines) {
            String trimmed = line.trim();
            if (trimmed.isEmpty()) continue;
            
            // 문제 번호와 답안이 함께 있는 경우 (예: "1: 3", "1번: 5")
            Matcher matcher = questionNumberPattern.matcher(trimmed);
            if (matcher.find()) {
                int questionNumber = Integer.parseInt(matcher.group(1));
                String answer = matcher.group(2);
                answerMap.put(questionNumber, answer);
                log.debug("Extracted question {}: answer {}", questionNumber, answer);
            } else {
                // 숫자만 포함된 라인 (문제 번호 없이 답안만)
                if (trimmed.matches("\\d+")) {
                    answerMap.put(sequentialIndex++, trimmed);
                } else if (trimmed.matches(".*\\d+.*")) {
                    // 숫자가 포함된 경우 숫자만 추출
                    String numbers = trimmed.replaceAll("[^0-9]", "");
                    if (!numbers.isEmpty()) {
                        answerMap.put(sequentialIndex++, numbers);
                    }
                }
            }
        }
        
        log.info("Extracted {} answers from image (with question numbers)", answerMap.size());
        return answerMap;
    }
    
    /**
     * 이미지에서 답안을 순서대로 추출합니다 (기존 방식, 하위 호환성 유지).
     * 
     * @param imageBytes 이미지 바이트 배열
     * @return 답안 리스트 (순서대로)
     */
    public List<String> extractAnswersFromImage(byte[] imageBytes) throws IOException {
        Map<Integer, String> answerMap = extractAnswersWithQuestionNumbers(imageBytes);
        // 문제 번호 순서대로 정렬하여 리스트로 변환
        return new ArrayList<>(answerMap.values());
    }
}

