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
     * 이미지에서 OCR 응답 추출 (텍스트와 confidence 정보 포함)
     */
    private AnnotateImageResponse extractOcrResponse(byte[] imageBytes) throws IOException {
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

            return res;

        } catch (Exception e) {
            log.error("OCR processing failed", e);
            throw new IOException("OCR failed", e);
        }
    }

    /**
     * 이미지에서 OCR 전체 텍스트 추출
     */
    private String extractFullText(byte[] imageBytes) throws IOException {
        AnnotateImageResponse res = extractOcrResponse(imageBytes);

        if (res.hasFullTextAnnotation()) {
            return res.getFullTextAnnotation().getText();
        }

        // fallback
        StringBuilder sb = new StringBuilder();
        for (EntityAnnotation annotation : res.getTextAnnotationsList()) {
            sb.append(annotation.getDescription()).append("\n");
        }

        return sb.toString();
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

    /**
     * 문제 번호를 정규화합니다.
     * "15-1.", "15-1)", "15-1번" 등을 모두 "15-1"로 통일합니다.
     * 단일 문제 번호("1", "1번", "1.")는 "1"로 정규화합니다.
     */
    private String normalizeQuestionNumber(String questionNumber) {
        if (questionNumber == null || questionNumber.isEmpty()) {
            return questionNumber;
        }
        
        String normalized = questionNumber.trim();
        
        // 끝의 구분자 제거 (번, ., ), : 등)
        normalized = normalized.replaceAll("[번:.)]\\s*$", "");
        
        return normalized;
    }

    /**
     * 줄이 문제 번호만 있는지 확인합니다.
     * 지원 형식: "1.", "14.", "15-1)", "16-2)" 등
     * 숫자만 있는 줄("0", "30" 등)은 문제 번호가 아닌 답으로 간주합니다.
     */
    private boolean isQuestionNumberLine(String line) {
        if (line == null || line.trim().isEmpty()) {
            return false;
        }
        
        String trimmed = line.trim();
        
        // 숫자만 있는 줄은 문제 번호가 아님 (답으로 간주)
        if (trimmed.matches("^\\d+$")) {
            return false;
        }
        
        // 소문제 번호 패턴을 먼저 검사: "15-1)", "16-2)", "15-1." 등
        if (trimmed.matches("^\\d{1,2}-\\d{1,2}[번.)]?\\s*$")) {
            return true;
        }
        
        // 단일 문제 번호 패턴: "1.", "14.", "1번" 등
        if (trimmed.matches("^\\d{1,2}[번.)]?\\s*$")) {
            return true;
        }
        
        return false;
    }

    /**
     * 문제 번호를 추출합니다.
     * "1.", "14.", "15-1)", "16-2)" 등에서 문제 번호를 추출하고 정규화합니다.
     * 소문제 패턴을 단일 문제보다 먼저 검사하여 "15-1)"이 "15"로 잘리는 버그를 방지합니다.
     */
    private String extractQuestionNumber(String line) {
        if (line == null || line.trim().isEmpty()) {
            return null;
        }
        
        String trimmed = line.trim();
        
        // 소문제 번호를 먼저 검사 (단일 문제보다 우선)
        // "15-1)"이 "15"로 잘리는 버그 방지
        Pattern subPattern = Pattern.compile("^(\\d{1,2}-\\d{1,2})[번.)]?");
        Matcher subMatcher = subPattern.matcher(trimmed);
        if (subMatcher.find()) {
            return normalizeQuestionNumber(subMatcher.group(1));
        }
        
        // 단일 문제 번호 추출: "1.", "14." 등
        Pattern singlePattern = Pattern.compile("^(\\d{1,2})[번.)]?");
        Matcher singleMatcher = singlePattern.matcher(trimmed);
        if (singleMatcher.find()) {
            return normalizeQuestionNumber(singleMatcher.group(1));
        }
        
        return null;
    }

    /**
     * 줄이 답안인지 확인합니다.
     * 답안 형식: 숫자, True/False, 한글 단어, 영어 단어
     * 문제 번호 패턴이 아닌 경우 답안으로 간주합니다.
     */
    private boolean isAnswerLine(String line) {
        if (line == null || line.trim().isEmpty()) {
            return false;
        }
        
        String trimmed = line.trim();
        
        // 문제 번호 패턴이면 답안이 아님
        if (isQuestionNumberLine(trimmed)) {
            return false;
        }
        
        // 숫자만 있는 경우 (답안으로 간주)
        if (trimmed.matches("^\\d+$")) {
            return true;
        }
        
        // Boolean 값
        if (trimmed.matches("^(?i)(true|false)$")) {
            return true;
        }
        
        // 한글이 포함된 경우 (한글 단어, 숫자+한글 조합 등)
        if (trimmed.matches(".*[가-힣]+.*")) {
            return true;
        }
        
        // 영어 단어 (알파벳만 또는 알파벳+숫자 조합)
        if (trimmed.matches("^[a-zA-Z0-9\\s]+$") && trimmed.length() > 0) {
            // 문제 번호 패턴이 아닌 경우만
            if (!trimmed.matches("^\\d{1,2}[번.)]?\\s*$") && 
                !trimmed.matches("^\\d{1,2}-\\d{1,2}[번.)]?\\s*$")) {
                return true;
            }
        }
        
        return false;
    }

    /**
     * 답안을 추출합니다.
     * 숫자, True/False, 한글/영어 단어를 그대로 반환합니다.
     */
    private String extractAnswer(String line) {
        if (line == null || line.trim().isEmpty()) {
            return null;
        }
        
        String trimmed = line.trim();
        
        // Boolean 값은 대소문자 구분 없이 처리
        if (trimmed.matches("^(?i)(true|false)$")) {
            return trimmed.substring(0, 1).toUpperCase() + trimmed.substring(1).toLowerCase();
        }
        
        // 그 외는 그대로 반환 (공백 제거)
        return trimmed;
    }

    /**
     * 텍스트에서 답안 추출 (한 줄 매칭 우선, 상태 기반 파싱 보조)
     * 
     * 파싱 우선순위:
     * 1. 한 줄 매칭 (최우선): "1. 4", "15-1) 빅데이터" 등 문제번호+답이 같은 줄
     * 2. 상태 기반 파싱: 문제 번호만 있는 줄 → 다음 줄에서 답 추출
     */
    private Map<String, String> extractAnswersFromText(String text) {
        String[] lines = text.split("\n");
        Map<String, String> answerMap = new LinkedHashMap<>();
        String currentQuestion = null;

        // 한 줄 매칭 패턴: 문제번호 + 답이 같은 줄
        // 예: "1. 4", "2. 2501", "15-1) 빅데이터", "16-2) 인과분석"
        Pattern singleLinePattern = Pattern.compile("^(\\d{1,2}-\\d{1,2}|\\d{1,2})[번.)]?\\s+(.+)$");

        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];
            String trimmed = line.trim();
            
            // 빈 줄은 건너뛰기
            if (trimmed.isEmpty()) {
                continue;
            }

            // 1️⃣ 한 줄 매칭 (최우선)
            Matcher singleLineMatcher = singleLinePattern.matcher(trimmed);
            if (singleLineMatcher.find()) {
                String questionNumRaw = singleLineMatcher.group(1);
                String answerRaw = singleLineMatcher.group(2).trim();
                
                // 문제 번호 정규화
                String questionNum = normalizeQuestionNumber(questionNumRaw);
                String answer = extractAnswer(answerRaw);
                
                if (questionNum != null && answer != null && !answer.isEmpty()) {
                    answerMap.put(questionNum, answer);
                    log.debug("Single-line match: question {} -> answer {} (from: '{}')", questionNum, answer, trimmed);
                    
                    // 한 줄 매칭이 성공했으므로 currentQuestion 초기화
                    currentQuestion = null;
                    continue;
                }
            }

            // 2️⃣ 상태 기반 파싱 (한 줄 매칭 실패 시)
            // 문제 번호만 있는 줄인지 확인
            if (isQuestionNumberLine(trimmed)) {
                String questionNum = extractQuestionNumber(trimmed);
                if (questionNum != null) {
                    // 이전 문제에 답이 없었던 경우 경고
                    if (currentQuestion != null) {
                        log.warn("Question {} found but no answer was mapped (new question: {})", 
                                currentQuestion, questionNum);
                    }
                    currentQuestion = questionNum;
                    log.debug("Found question number: {} (from: '{}')", questionNum, trimmed);
                    continue;
                }
            }

            // 현재 문제 번호가 있고, 이 줄이 답안인 경우
            if (currentQuestion != null && isAnswerLine(trimmed)) {
                String answer = extractAnswer(trimmed);
                if (answer != null) {
                    answerMap.put(currentQuestion, answer);
                    log.debug("State-based match: question {} -> answer {} (from: '{}')", currentQuestion, answer, trimmed);
                    currentQuestion = null; // 답을 매핑한 뒤 초기화
                    continue;
                }
            }

            // 문제 번호가 설정되어 있지만 답안이 아닌 줄을 만난 경우
            // currentQuestion은 유지하여 다음 줄에서 답안을 찾음
            if (currentQuestion != null) {
                log.debug("Skipping non-answer line for question {}: '{}' (waiting for answer)", currentQuestion, trimmed);
            }
        }

        // 마지막에 currentQuestion이 남아있지만 답이 없는 경우 로그
        if (currentQuestion != null) {
            log.warn("Question {} found at the end but no answer was mapped", currentQuestion);
        }

        log.info("Extracted {} answers from text (single-line + stateful parsing)", answerMap.size());
        return answerMap;
    }

    /**
     * TextAnnotation에서 문제별 confidence 추출
     * Google Cloud Vision API는 Block 레벨에서 confidence를 제공하지 않으므로,
     * 전체 이미지에 대해 기본 confidence 값(0.95)을 사용합니다.
     */
    private Map<String, Float> extractConfidenceMap(
            TextAnnotation textAnnotation,
            Map<String, String> answers) {
        Map<String, Float> confidenceMap = new LinkedHashMap<>();

        if (textAnnotation == null) {
            return confidenceMap;
        }

        // Google Cloud Vision API의 TEXT_DETECTION은 Block 레벨 confidence를 제공하지 않음
        // 대신 기본 confidence 값 사용 (실제로는 OCR 품질에 따라 달라질 수 있음)
        // 일반적으로 Google Cloud Vision API의 텍스트 인식 정확도는 높으므로 0.95 사용
        float defaultConfidence = 0.95f;

        // 각 문제에 대해 기본 confidence 할당
        for (String questionNum : answers.keySet()) {
            confidenceMap.put(questionNum, defaultConfidence);
        }

        return confidenceMap;
    }

    // 최종 메서드
    public OcrResult extractStudentIdAndAnswers(
            byte[] imageBytes,
            String fileNameFallback
    ) throws IOException {

        AnnotateImageResponse ocrResponse = extractOcrResponse(imageBytes);
        String fullText = ocrResponse.hasFullTextAnnotation() 
                ? ocrResponse.getFullTextAnnotation().getText()
                : extractFullText(imageBytes);

        String studentId = extractStudentIdFromText(fullText);
        if (studentId == null) {
            studentId = fileNameFallback;
            log.warn("StudentId not found by OCR. Fallback to filename: {}", studentId);
        } else {
            log.info("Extracted studentId by OCR: {}", studentId);
        }

        Map<String, String> answers = extractAnswersFromText(fullText);
        Map<String, Float> confidenceMap = ocrResponse.hasFullTextAnnotation()
                ? extractConfidenceMap(ocrResponse.getFullTextAnnotation(), answers)
                : Map.of();

        log.info("Extracted {} answers", answers.size());

        return new OcrResult(studentId, answers, confidenceMap);
    }

    // 정답지 추출용. 학번 추출 안함
    public Map<String, String> extractAnswersOnly(byte[] imageBytes) throws IOException {

        String fullText = extractFullText(imageBytes);

        Map<String, String> answers = extractAnswersFromText(fullText);

        log.info("Extracted {} correct answers from answer sheet", answers.size());

        return answers;
    }

    /**
     * 정답지에서 답안과 confidence 정보를 함께 추출
     */
    public OcrResult extractAnswersWithConfidence(byte[] imageBytes) throws IOException {
        AnnotateImageResponse ocrResponse = extractOcrResponse(imageBytes);
        String fullText = ocrResponse.hasFullTextAnnotation() 
                ? ocrResponse.getFullTextAnnotation().getText()
                : extractFullText(imageBytes);

        Map<String, String> answers = extractAnswersFromText(fullText);
        Map<String, Float> confidenceMap = ocrResponse.hasFullTextAnnotation()
                ? extractConfidenceMap(ocrResponse.getFullTextAnnotation(), answers)
                : Map.of();

        log.info("Extracted {} correct answers from answer sheet", answers.size());

        return new OcrResult(null, answers, confidenceMap);
    }
}

