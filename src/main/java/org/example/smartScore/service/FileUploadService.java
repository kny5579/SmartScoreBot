package org.example.smartScore.service;

import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
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
        List<String> correctAnswers = ocrService.extractAnswersFromImage(answerImageBytes);
        log.info("Extracted {} correct answers from answer sheet", correctAnswers.size());

        // 학생 답안 이미지 처리
        List<GradingResult> gradingResults = new ArrayList<>();
        List<ImageFile> studentImages = new ArrayList<>();
        
        for (MultipartFile studentFile : studentFiles) {
            try {
                byte[] studentImageBytes = studentFile.getBytes();
                
                // OCR로 학생 답안 추출
                List<String> studentAnswers = ocrService.extractAnswersFromImage(studentImageBytes);
                
                // 파일명에서 학번 추출 (예: "2024123456.jpg" -> "2024123456")
                String fileName = studentFile.getOriginalFilename();
                String studentId = extractStudentId(fileName);
                
                log.info("Processing student {}: extracted {} answers", studentId, studentAnswers.size());
                
                // 채점 수행
                GradingResult result = gradingService.gradeAnswers(
                        studentAnswers, correctAnswers, studentId, examDate, userEmail);
                gradingResults.add(result);
                
                // 이미지 정보 저장 (나중에 excelId 설정)
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
        
        // 파일명에서 학번 추출 시도
        Matcher matcher = STUDENT_ID_PATTERN.matcher(fileName);
        if (matcher.find()) {
            return matcher.group(1);
        }
        
        // 확장자 제거 후 파일명 사용
        String nameWithoutExt = fileName.substring(0, fileName.lastIndexOf('.'));
        return nameWithoutExt.isEmpty() ? "unknown_" + System.currentTimeMillis() : nameWithoutExt;
    }

}

