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
     * 답안 보정
     * 답안을 추출합니다.
     * 숫자, True/False, 한글/영어 단어를 그대로 반환
     */
    private String extractAnswer(String line) {
        if (line == null || line.trim().isEmpty()) {
            return null;
        }

        String s = line.trim();

        if (s.isEmpty()) return null;

        return s;
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

    // Confidence 재계산 기준값 (threshold)
    private static final float CONFIDENCE_THRESHOLD = 0.7f;
    
    // 기본 OCR confidence (Google Cloud Vision API 기본값)
    private static final float BASE_OCR_CONFIDENCE = 0.95f;
    
    /**
     * TextAnnotation에서 문제별 confidence 추출 및 재계산
     * 
     * 재계산 기준:
     * - Base Score: 기본 OCR confidence (0.95, 가중치 0.3)
     * - Length Score: 답안 길이 기반 (가중치 0.2)
     * - Regex Score: 정규식 매칭 여부 (가중치 0.3)
     * - Type Score: 타입 변환 성공 여부 (가중치 0.2)
     * 
     * @param textAnnotation TextAnnotation 객체
     * @param answers 추출된 답안 맵
     * @return ConfidenceResult 객체 (confidence 맵과 lowConfidence 맵 포함)
     */
    private ConfidenceResult extractConfidenceMap(
            TextAnnotation textAnnotation,
            Map<String, String> answers) {
        Map<String, Float> confidenceMap = new LinkedHashMap<>();
        Map<String, Boolean> lowConfidenceMap = new LinkedHashMap<>();

        if (textAnnotation == null || answers == null || answers.isEmpty()) {
            return new ConfidenceResult(confidenceMap, lowConfidenceMap);
        }

        // 각 문제에 대해 동적으로 confidence 재계산
        for (Map.Entry<String, String> entry : answers.entrySet()) {
            String questionNum = entry.getKey();
            String answer = entry.getValue();
            
            // 빈 답안 처리 (null, 빈 문자열, 공백)
            if (answer == null || answer.trim().isEmpty()) {
                confidenceMap.put(questionNum, 0.0f);
                lowConfidenceMap.put(questionNum, true);
                log.warn("Empty answer for question {}: confidence=0.0, lowConfidence=true", questionNum);
                continue;
            }
            
            // 동적 confidence 재계산
            ConfidenceScores scores = recalculateConfidence(questionNum, answer);
            float finalConfidence = scores.finalConfidence;
            confidenceMap.put(questionNum, finalConfidence);
            
            // threshold 이하인 경우 lowConfidence 플래그 설정
            boolean isLowConfidence = finalConfidence <= CONFIDENCE_THRESHOLD;
            lowConfidenceMap.put(questionNum, isLowConfidence);
            
            // 디버깅 로그
            log.debug("Question {}: baseScore={}, lengthScore={}, regexScore={}, typeScore={}, finalConfidence={}, answer='{}'", 
                    questionNum, 
                    String.format("%.2f", scores.baseScore),
                    String.format("%.2f", scores.lengthScore),
                    String.format("%.2f", scores.regexScore),
                    String.format("%.2f", scores.typeScore),
                    String.format("%.2f", finalConfidence),
                    answer);
            
            if (isLowConfidence) {
                log.warn("Low confidence detected for question {}: confidence={}, answer='{}'", 
                        questionNum, String.format("%.2f", finalConfidence), answer);
            }
        }

        return new ConfidenceResult(confidenceMap, lowConfidenceMap);
    }
    
    /**
     * Confidence 점수들을 담는 내부 클래스
     */
    private static class ConfidenceScores {
        final float baseScore;
        final float lengthScore;
        final float regexScore;
        final float typeScore;
        final float finalConfidence;
        
        ConfidenceScores(float baseScore, float lengthScore, float regexScore, float typeScore, float finalConfidence) {
            this.baseScore = baseScore;
            this.lengthScore = lengthScore;
            this.regexScore = regexScore;
            this.typeScore = typeScore;
            this.finalConfidence = finalConfidence;
        }
    }
    
    /**
     * Confidence 재계산
     * 
     * 가중치: base 0.3, length 0.2, regex 0.3, type 0.2 (합계 1.0)
     * 
     * @param questionNum 문제 번호
     * @param answer 답안 (null이 아니고 빈 문자열이 아님을 보장)
     * @return ConfidenceScores 객체 (각 점수와 최종 confidence 포함)
     */
    private ConfidenceScores recalculateConfidence(String questionNum, String answer) {
        // 1. Base Score (기본 OCR confidence)
        float baseScore = BASE_OCR_CONFIDENCE;
        
        // 2. Length Score (문자 길이 기반)
        float lengthScore = calculateLengthScore(answer);
        
        // 3. Regex Score (정규식 매칭 기반)
        float regexScore = calculateRegexScore(questionNum, answer);
        
        // 4. Type Score (타입 변환 성공 여부)
        float typeScore = calculateTypeScore(answer);
        
        // 가중 평균으로 최종 confidence 계산 (합계 1.0)
        // base: 0.3, length: 0.2, regex: 0.3, type: 0.2
        float finalConfidence = (baseScore * 0.3f) + 
                                (lengthScore * 0.2f) + 
                                (regexScore * 0.3f) + 
                                (typeScore * 0.2f);
        
        // 0.0 ~ 1.0 범위로 제한
        finalConfidence = Math.max(0.0f, Math.min(1.0f, finalConfidence));
        
        return new ConfidenceScores(baseScore, lengthScore, regexScore, typeScore, finalConfidence);
    }
    
    /**
     * 문자 길이 기반 점수 계산
     * 
     * @param answer 답안
     * @return 길이 기반 점수 (0.0 ~ 1.0)
     */
    private float calculateLengthScore(String answer) {
        if (answer == null || answer.trim().isEmpty()) {
            return 0.0f;
        }
        
        int length = answer.trim().length();
        
        // 답안 길이가 너무 짧거나 길면 점수 감소
        if (length == 0) {
            return 0.0f;
        } else if (length == 1) {
            return 0.8f;  // 단일 문자 답안 (예: "1", "O", "X")
        } else if (length >= 2 && length <= 10) {
            return 1.0f;  // 적절한 길이
        } else if (length > 10 && length <= 20) {
            return 0.9f;  // 다소 긴 답안
        } else {
            return 0.7f;  // 매우 긴 답안 (의심스러움)
        }
    }
    
    /**
     * 정규식 매칭 여부 기반 점수 계산
     * 
     * 숫자형 문제에서 "|", "나", "l", "I" 등 숫자가 아닌 문자는 0점
     * 
     * @param questionNum 문제 번호
     * @param answer 답안
     * @return 정규식 매칭 기반 점수 (0.0 ~ 1.0)
     */
    private float calculateRegexScore(String questionNum, String answer) {
        String trimmed = answer.trim();
        
        // 숫자만 있는 답안 (숫자형) - 순수 숫자만 허용
        if (trimmed.matches("^\\d+$")) {
            return 1.0f;
        }
        
        // Boolean 값 패턴 - true/false만 허용 (대소문자 무관)
        if (trimmed.matches("^(?i)(true|false)$")) {
            return 1.0f;
        }
        
        // 숫자형으로 보이지만 오인식 문자가 포함된 경우 (|, 나, l, I 등)
        // 숫자가 아닌 문자가 포함되어 있으면 0점
        if (trimmed.matches(".*[|나lI!].*")) {
            // 숫자로 변환 시도
            String numericResult = tryNumericConversion(trimmed);
            if (numericResult == null || numericResult.isEmpty()) {
                return 0.0f;  // 숫자로 변환 실패
            }
            // 숫자로 변환은 되지만 원본에 오인식 문자가 있으면 점수 감소
            if (!trimmed.equals(numericResult)) {
                return 0.0f;  // 오인식 문자 포함
            }
        }
        
        // 한글이 포함된 답안
        if (trimmed.matches(".*[가-힣]+.*")) {
            return 0.8f;
        }
        
        // 영어 단어 답안 (알파벳과 숫자 조합)
        if (trimmed.matches("^[a-zA-Z0-9\\s]+$")) {
            return 0.8f;
        }
        
        // 특수 문자만 있는 경우
        if (trimmed.matches("^[^a-zA-Z0-9가-힣]+$")) {
            return 0.0f;
        }
        
        // 그 외 (혼합)
        return 0.5f;
    }
    
    /**
     * 타입 변환 성공 여부 기반 점수 계산
     * 
     * - 숫자형: 숫자로 정상 파싱된 경우만 1.0, 오인식 문자 포함 시 0.0
     * - Boolean: true/false만 허용, 그 외 0.0
     * - 문자열: 기본 점수
     * 
     * @param answer 답안
     * @return 타입 변환 기반 점수 (0.0 ~ 1.0)
     */
    private float calculateTypeScore(String answer) {
        String trimmed = answer.trim();
        
        // 숫자형 변환 시도
        String numericResult = tryNumericConversion(trimmed);
        if (numericResult != null && !numericResult.isEmpty()) {
            // 원본과 변환 결과가 다르면 오인식 문자 포함
            if (!trimmed.equals(numericResult)) {
                // 오인식 문자(|, 나, l, I 등)가 포함된 경우
                return 0.0f;
            }
            // 순수 숫자로 정상 파싱된 경우
            return 1.0f;
        }
        
        // Boolean 변환 시도
        String booleanResult = tryBooleanConversion(trimmed);
        if (booleanResult != null) {
            // true/false만 허용 (대소문자 무관)
            if (trimmed.matches("^(?i)(true|false)$")) {
                return 1.0f;
            }
            // 그 외 Boolean 변환 가능한 값 (1, 0, O, X 등)은 0점
            return 0.0f;
        }
        
        // 문자열 타입 (기본 점수)
        return 0.5f;
    }
    
    /**
     * 숫자형 변환 시도
     * 
     * @param answer 답안
     * @return 변환된 숫자 문자열, 실패 시 null
     */
    private String tryNumericConversion(String answer) {
        if (answer == null || answer.trim().isEmpty()) {
            return null;
        }
        
        String corrected = answer.trim();
        
        // 특수 문자 보정 (GradingService와 동일한 로직)
        corrected = corrected.replace("나", "4");
        corrected = corrected.replace("|", "1");
        corrected = corrected.replace("Ⅰ", "1");
        corrected = corrected.replace("l", "1");
        corrected = corrected.replace("!", "1");
        corrected = corrected.replace("O", "0");
        corrected = corrected.replace("o", "0");
        corrected = corrected.replace("S", "5");
        corrected = corrected.replace("B", "8");
        
        // 숫자가 아닌 문자 제거
        corrected = corrected.replaceAll("[^0-9]", "");
        
        if (corrected.isEmpty()) {
            return null;
        }
        
        return corrected;
    }
    
    /**
     * Boolean 변환 시도
     * 
     * true/false만 허용 (대소문자 무관)
     * 
     * @param answer 답안
     * @return 변환된 Boolean 문자열 ("true" 또는 "false"), 실패 시 null
     */
    private String tryBooleanConversion(String answer) {
        String trimmed = answer.trim();
        String lower = trimmed.toLowerCase();
        
        // true/false만 허용
        if (lower.equals("true")) {
            return "true";
        }
        
        if (lower.equals("false")) {
            return "false";
        }
        
        return null;
    }
    
    /**
     * Confidence 결과를 담는 내부 클래스
     */
    private static class ConfidenceResult {
        final Map<String, Float> confidenceMap;
        final Map<String, Boolean> lowConfidenceMap;
        
        ConfidenceResult(Map<String, Float> confidenceMap, Map<String, Boolean> lowConfidenceMap) {
            this.confidenceMap = confidenceMap;
            this.lowConfidenceMap = lowConfidenceMap;
        }
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
        ConfidenceResult confidenceResult = ocrResponse.hasFullTextAnnotation()
                ? extractConfidenceMap(ocrResponse.getFullTextAnnotation(), answers)
                : new ConfidenceResult(Map.of(), Map.of());

        log.info("Extracted {} answers", answers.size());

        return new OcrResult(studentId, answers, 
                confidenceResult.confidenceMap, 
                confidenceResult.lowConfidenceMap);
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
        ConfidenceResult confidenceResult = ocrResponse.hasFullTextAnnotation()
                ? extractConfidenceMap(ocrResponse.getFullTextAnnotation(), answers)
                : new ConfidenceResult(Map.of(), Map.of());

        log.info("Extracted {} correct answers from answer sheet", answers.size());

        return new OcrResult(null, answers, 
                confidenceResult.confidenceMap, 
                confidenceResult.lowConfidenceMap);
    }

}

