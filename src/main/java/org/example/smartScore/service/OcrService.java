package org.example.smartScore.service;

import com.google.cloud.vision.v1.*;
import com.google.protobuf.ByteString;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.dto.OcrResult;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Service
@RequiredArgsConstructor
@org.springframework.boot.autoconfigure.condition.ConditionalOnBean(ImageAnnotatorClient.class)
public class OcrService {

    private final ImageAnnotatorClient imageAnnotatorClient;

    /**
     * 이미지에서 OCR 전체 텍스트 추출
     */
    private String extractFullText(byte[] imageBytes) throws IOException {
        try {
            ByteString imgBytes = ByteString.copyFrom(imageBytes);

            Image img = Image.newBuilder().setContent(imgBytes).build();
            Feature feature = Feature.newBuilder()
                    .setType(Feature.Type.TEXT_DETECTION)
                    .build();

            AnnotateImageRequest request = AnnotateImageRequest.newBuilder()
                    .setImage(img)
                    .addFeatures(feature)
                    .build();

            BatchAnnotateImagesResponse response =
                    imageAnnotatorClient.batchAnnotateImages(List.of(request));

            AnnotateImageResponse res = response.getResponsesList().get(0);

            if (res.hasError()) {
                throw new IOException(res.getError().getMessage());
            }

            if (res.hasFullTextAnnotation()) {
                return res.getFullTextAnnotation().getText();
            }

            // fallback
            StringBuilder sb = new StringBuilder();
            for (EntityAnnotation annotation : res.getTextAnnotationsList()) {
                sb.append(annotation.getDescription()).append("\n");
            }

            return sb.toString();

        } catch (Exception e) {
            log.error("OCR processing failed", e);
            throw new IOException("OCR failed", e);
        }
    }

    // OCR 텍스트에서 학번 추출
    private String extractStudentIdFromText(String text) {
        // 예: 학번: 20231234 / 20231234
        Pattern pattern = Pattern.compile("(학번\\s*[:：]?\\s*)?(\\d{8})");
        Matcher matcher = pattern.matcher(text);

        if (matcher.find()) {
            return matcher.group(2);
        }
        return null;
    }

    // 텍스트에서 답안 추출 (문제번호 포함)
    private Map<Integer, String> extractAnswersFromText(String text) {
        String[] lines = text.split("\n");
        Map<Integer, String> answerMap = new LinkedHashMap<>();
        int seq = 1;

        Pattern questionPattern = Pattern.compile("(\\d+)[번:.]?\\s*([0-9]+)");

        for (String line : lines) {
            String trimmed = line.trim();
            if (trimmed.isEmpty()) continue;

            Matcher matcher = questionPattern.matcher(trimmed);
            if (matcher.find()) {
                int qNum = Integer.parseInt(matcher.group(1));
                String answer = matcher.group(2);
                answerMap.put(qNum, answer);
            } else if (trimmed.matches("\\d+")) {
                answerMap.put(seq++, trimmed);
            }
        }

        return answerMap;
    }

    // 최종 메서드
    public OcrResult extractStudentIdAndAnswers(
            byte[] imageBytes,
            String fileNameFallback
    ) throws IOException {

        String fullText = extractFullText(imageBytes);

        String studentId = extractStudentIdFromText(fullText);
        if (studentId == null) {
            studentId = fileNameFallback;
            log.warn("StudentId not found by OCR. Fallback to filename: {}", studentId);
        } else {
            log.info("Extracted studentId by OCR: {}", studentId);
        }

        Map<Integer, String> answers = extractAnswersFromText(fullText);

        log.info("Extracted {} answers", answers.size());

        return new OcrResult(studentId, answers);
    }

    // 정답지 추출용. 학번 추출 안함
    public Map<Integer, String> extractAnswersOnly(byte[] imageBytes) throws IOException {

        String fullText = extractFullText(imageBytes);

        Map<Integer, String> answers = extractAnswersFromText(fullText);

        log.info("Extracted {} correct answers from answer sheet", answers.size());

        return answers;
    }
}
