package org.example.smartScore.service;

import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.dto.OcrResult;
import org.example.smartScore.repository.ExcelFileRepository;
import org.example.smartScore.repository.ImageFileRepository;
import org.example.smartScore.service.GradingService.GradingResult;
import org.example.smartScore.util.DateUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.sql.Timestamp;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Service
@Transactional
public class FileUploadService {

    private final ExcelFileRepository excelFileRepository;
    private final ImageFileRepository imageFileRepository;
    private final GradingService gradingService;
    private final ExcelGenerationService excelGenerationService;
    
    // Optional로 변경하여 OCR이 없어도 애플리케이션이 시작되도록
    @Autowired(required = false)
    private OcrService ocrService;
    
    public FileUploadService(ExcelFileRepository excelFileRepository,
                            ImageFileRepository imageFileRepository,
                            GradingService gradingService,
                            ExcelGenerationService excelGenerationService) {
        this.excelFileRepository = excelFileRepository;
        this.imageFileRepository = imageFileRepository;
        this.gradingService = gradingService;
        this.excelGenerationService = excelGenerationService;
    }
    
    private static final Pattern STUDENT_ID_PATTERN = Pattern.compile("(\\d{8,10})"); // 학번 패턴 (8-10자리 숫자)

    public String uploadAndProcessFiles(MultipartFile[] studentFiles, MultipartFile[] answerFiles, 
                                       String dateString, String userEmail) throws IOException, ParseException {
        
        log.info("File upload started for user: {}, exam date: {}", userEmail, dateString);
        
        // OCR 서비스가 없는 경우 에러 반환
        if (ocrService == null) {
            throw new IllegalStateException("OCR 기능이 활성화되지 않았습니다. application.yml에서 google.cloud.vision.enabled=true로 설정하고 인증 정보를 구성해주세요.");
        }
        
        Date examDate = DateUtils.parseDate(dateString);
        Timestamp submitDate = new Timestamp(System.currentTimeMillis());

        // 정답 이미지에서 답안 추출 (첫 번째 정답 이미지 사용)
        if (answerFiles == null || answerFiles.length == 0) {
            throw new IllegalArgumentException("정답 이미지가 필요합니다.");
        }
        
        byte[] answerImageBytes = answerFiles[0].getBytes();
        // 답안 추출 (문제 번호 기준 매핑, confidence 포함)
        OcrResult answerOcrResult = ocrService.extractAnswersWithConfidence(answerImageBytes);
        Map<String, String> correctAnswersMap = answerOcrResult.answers();
        Map<String, Float> answerConfidenceMap = answerOcrResult.confidenceMap();
        log.info("Extracted {} correct answers from answer sheet", correctAnswersMap.size());
        
        // 정답 이미지의 confidence 통계 로그
        if (!answerConfidenceMap.isEmpty()) {
            double avgAnswerConfidence = answerConfidenceMap.values().stream()
                    .mapToDouble(Float::doubleValue)
                    .average()
                    .orElse(0.0);
            log.info("정답 이미지 평균 Confidence: {}%", String.format("%.2f", avgAnswerConfidence * 100));
        }

        // 학생 답안 이미지 처리
        List<GradingResult> gradingResults = new ArrayList<>();
        List<ImageFile> studentImages = new ArrayList<>();
        
        // OCR 정확도 통계 수집용
        Map<String, List<Float>> questionConfidenceMap = new LinkedHashMap<>(); // 문제별 confidence 리스트
        List<OcrResult> ocrResults = new ArrayList<>(); // 이미지별 OCR 결과

        for (MultipartFile studentFile : studentFiles) {
            try {
                byte[] studentImageBytes = studentFile.getBytes();

                String fileName = studentFile.getOriginalFilename();
                String fallbackStudentId = extractStudentId(fileName);

                OcrResult ocrResult = ocrService.extractStudentIdAndAnswers(
                        studentImageBytes,
                        fallbackStudentId
                );

                String studentId = ocrResult.studentId();
                Map<String, String> studentAnswersMap = ocrResult.answers();
                Map<String, Float> confidenceMap = ocrResult.confidenceMap();

                // 문제별 confidence 수집
                for (Map.Entry<String, Float> entry : confidenceMap.entrySet()) {
                    questionConfidenceMap.computeIfAbsent(entry.getKey(), k -> new ArrayList<>())
                            .add(entry.getValue());
                }

                ocrResults.add(ocrResult);

                log.info("Processing student {}: extracted {} answers",
                        studentId, studentAnswersMap.size());

                GradingResult result =
                        gradingService.gradeAnswersWithQuestionNumbers(
                                studentAnswersMap,
                                correctAnswersMap,
                                studentId,
                                examDate,
                                userEmail
                        );

                gradingResults.add(result);

                ImageFile imageFile = new ImageFile();
                imageFile.setImageName(fileName);
                imageFile.setData(studentImageBytes);
                imageFile.setExamDate(examDate);
                imageFile.setSubmitDate(submitDate);
                imageFile.setEmail(userEmail);

                studentImages.add(imageFile);

            } catch (Exception e) {
                log.error("Error processing student file: {}", studentFile.getOriginalFilename(), e);
                throw new RuntimeException("학생 답안 처리 중 오류 발생: " + studentFile.getOriginalFilename(), e);
            }
        }

        // OCR 정확도 로그 시각화
        logOcrAccuracyStatistics(questionConfidenceMap, ocrResults, gradingResults, correctAnswersMap);

        // 채점 결과를 Excel 파일로 생성
        byte[] excelData = excelGenerationService.generateGradingExcel(gradingResults, examDate);
        String excelFileName = "grading_result_" + new SimpleDateFormat("yyyyMMdd_HHmmss").format(submitDate) + ".xlsx";
        
        // Excel 파일 저장
        ExcelFile excelFile = new ExcelFile();
        excelFile.setFileName(excelFileName);
        excelFile.setData(excelData);
        excelFile.setExamDate(examDate);
        excelFile.setSubmitDate(submitDate);
        excelFile.setEmail(userEmail);
        ExcelFile savedExcelFile = excelFileRepository.save(excelFile);
        Long excelFileId = savedExcelFile.getId();
        
        // 학생 이미지들을 Excel 파일과 연결하여 저장
        for (ImageFile imageFile : studentImages) {
            imageFile.setExcelId(excelFileId);
            imageFileRepository.save(imageFile);
        }
        
        log.info("File upload completed successfully for user: {}, processed {} students", 
                userEmail, gradingResults.size());
        return "success";
    }

    private String extractStudentId(String fileName) {
        if (fileName == null || fileName.isEmpty()) {
            return "unknown_" + System.currentTimeMillis();
        }

        Matcher matcher = STUDENT_ID_PATTERN.matcher(fileName);
        if (matcher.find()) {
            return matcher.group(1);
        }

        // 확장자 제거
        int dotIndex = fileName.lastIndexOf('.');
        return dotIndex > 0 ? fileName.substring(0, dotIndex) : fileName;
    }

    /**
     * 문제 번호를 정렬하기 위한 비교 함수 (소문제 지원)
     * "1", "2", "15-1", "15-2", "16" 순서로 정렬
     */
    private int compareQuestionNumbers(String q1, String q2) {
        // 소문제가 없는 경우와 있는 경우를 구분
        boolean q1HasSub = q1.contains("-");
        boolean q2HasSub = q2.contains("-");
        
        if (!q1HasSub && !q2HasSub) {
            // 둘 다 단일 문제: 숫자로 비교
            try {
                return Integer.compare(Integer.parseInt(q1), Integer.parseInt(q2));
            } catch (NumberFormatException e) {
                return q1.compareTo(q2);
            }
        }
        
        if (q1HasSub && q2HasSub) {
            // 둘 다 소문제: 메인 문제 번호 먼저 비교, 같으면 소문제 번호 비교
            String[] parts1 = q1.split("-");
            String[] parts2 = q2.split("-");
            if (parts1.length == 2 && parts2.length == 2) {
                try {
                    int main1 = Integer.parseInt(parts1[0]);
                    int main2 = Integer.parseInt(parts2[0]);
                    int cmp = Integer.compare(main1, main2);
                    if (cmp != 0) return cmp;
                    
                    int sub1 = Integer.parseInt(parts1[1]);
                    int sub2 = Integer.parseInt(parts2[1]);
                    return Integer.compare(sub1, sub2);
                } catch (NumberFormatException e) {
                    return q1.compareTo(q2);
                }
            }
        }
        
        // 하나는 소문제, 하나는 단일 문제: 메인 번호로 비교
        String main1 = q1HasSub ? q1.split("-")[0] : q1;
        String main2 = q2HasSub ? q2.split("-")[0] : q2;
        try {
            int cmp = Integer.compare(Integer.parseInt(main1), Integer.parseInt(main2));
            if (cmp != 0) return cmp;
            // 메인 번호가 같으면 소문제가 있는 것이 뒤로
            return q1HasSub ? 1 : -1;
        } catch (NumberFormatException e) {
            return q1.compareTo(q2);
        }
    }

    /**
     * OCR 정확도 통계를 로그로 출력
     * - 문제별 confidence 평균
     * - 이미지별 인식 성공률
     * - 문제별 인식 결과, 정답, 일치 여부
     */
    private void logOcrAccuracyStatistics(
            Map<String, List<Float>> questionConfidenceMap,
            List<OcrResult> ocrResults,
            List<GradingResult> gradingResults,
            Map<String, String> correctAnswersMap) {
        
        log.info("========== OCR 정확도 통계 ==========");
        
        // 문제별 confidence 평균 계산 및 출력
        if (!questionConfidenceMap.isEmpty()) {
            log.info("--- 문제별 Confidence 평균 ---");
            List<String> sortedQuestions = new ArrayList<>(questionConfidenceMap.keySet());
            sortedQuestions.sort(this::compareQuestionNumbers);
            
            for (String questionNum : sortedQuestions) {
                List<Float> confidences = questionConfidenceMap.get(questionNum);
                double avgConfidence = confidences.stream()
                        .mapToDouble(Float::doubleValue)
                        .average()
                        .orElse(0.0);
                
                log.info("문제 {}: 평균 Confidence = {}% (샘플 수: {})", 
                        questionNum, String.format("%.2f", avgConfidence * 100), confidences.size());
            }
        }
        
        // 이미지별 상세 정보 출력 (인식 결과, 정답, 일치 여부)
        if (!ocrResults.isEmpty() && !gradingResults.isEmpty()) {
            log.info("--- 이미지별 상세 인식 결과 ---");
            int totalQuestions = correctAnswersMap.size();
            
            for (int i = 0; i < ocrResults.size(); i++) {
                OcrResult ocrResult = ocrResults.get(i);
                GradingResult gradingResult = i < gradingResults.size() ? gradingResults.get(i) : null;
                
                Map<String, String> extractedAnswers = ocrResult.answers();
                Map<String, Boolean> questionResults = gradingResult != null 
                        ? gradingResult.questionResults() 
                        : Collections.emptyMap();
                
                // 인식된 문제 수 계산
                int recognizedCount = 0;
                for (String questionNum : correctAnswersMap.keySet()) {
                    if (extractedAnswers.containsKey(questionNum)) {
                        recognizedCount++;
                    }
                }
                
                // 인식 성공률 계산
                double recognitionRate = totalQuestions > 0 
                        ? (double) recognizedCount / totalQuestions * 100.0 
                        : 0.0;
                
                // 평균 confidence 계산
                Map<String, Float> confidenceMap = ocrResult.confidenceMap();
                double avgConfidence = confidenceMap.isEmpty() 
                        ? 0.0
                        : confidenceMap.values().stream()
                                .mapToDouble(Float::doubleValue)
                                .average()
                                .orElse(0.0);
                
                String studentId = ocrResult.studentId() != null 
                        ? ocrResult.studentId() 
                        : "Unknown";
                
                log.info("이미지 #{} (학번: {}): 인식 성공률 = {}% ({}/{}), 평균 Confidence = {}%", 
                        i + 1, studentId, 
                        String.format("%.2f", recognitionRate), 
                        recognizedCount, totalQuestions, 
                        String.format("%.2f", avgConfidence * 100));
                
                // 문제별 상세 정보 출력
                List<String> sortedQuestionNums = new ArrayList<>(correctAnswersMap.keySet());
                sortedQuestionNums.sort(this::compareQuestionNumbers);
                
                log.info("  [문제별 상세 결과]");
                for (String questionNum : sortedQuestionNums) {
                    String recognizedAnswer = extractedAnswers.getOrDefault(questionNum, "(미인식)");
                    String correctAnswer = correctAnswersMap.getOrDefault(questionNum, "(정답없음)");
                    boolean isCorrect = questionResults.getOrDefault(questionNum, false);
                    Float confidence = confidenceMap.getOrDefault(questionNum, 0.0f);
                    
                    String matchStatus = isCorrect ? "일치" : "불일치";
                    log.info("    문제 {}: 인식={}, 정답={}, 일치여부={}, Confidence={}%", 
                            questionNum, recognizedAnswer, correctAnswer, matchStatus,
                            String.format("%.2f", confidence * 100));
                }
            }
            
            // 전체 평균 인식 성공률
            double overallRecognitionRate = ocrResults.stream()
                    .mapToDouble(result -> {
                        Map<String, String> answers = result.answers();
                        long recognized = correctAnswersMap.keySet().stream()
                                .filter(answers::containsKey)
                                .count();
                        return totalQuestions > 0 ? (double) recognized / totalQuestions * 100.0 : 0.0;
                    })
                    .average()
                    .orElse(0.0);
            
            log.info("--- 전체 평균 인식 성공률: {}% ---", String.format("%.2f", overallRecognitionRate));
        }
        
        log.info("====================================");
    }

}

