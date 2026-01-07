package org.example.smartScore.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.constants.AppConstants;
import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.repository.ExcelFileRepository;
import org.example.smartScore.repository.ImageFileRepository;
import org.example.smartScore.util.DateUtils;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.sql.Timestamp;
import java.text.ParseException;
import java.util.Date;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional
public class FileUploadService {

    private final ExcelFileRepository excelFileRepository;
    private final ImageFileRepository imageFileRepository;
    private final RestTemplate restTemplate;

    public String uploadAndProcessFiles(MultipartFile[] studentFiles, MultipartFile[] answerFiles, 
                                       String dateString, String userEmail) throws IOException, ParseException {
        
        log.info("File upload started for user: {}", userEmail);
        
        Date examDate = DateUtils.parseDate(dateString);
        Timestamp submitDate = new Timestamp(System.currentTimeMillis());

        byte[] zipFileBytes = sendFilesToFlaskServer(studentFiles, answerFiles, examDate);
        
        if (zipFileBytes == null) {
            throw new IllegalStateException("Flask server returned empty response");
        }

        processZipFile(zipFileBytes, examDate, submitDate, userEmail);
        
        log.info("File upload completed successfully for user: {}", userEmail);
        return "success";
    }

    private byte[] sendFilesToFlaskServer(MultipartFile[] studentFiles, MultipartFile[] answerFiles, Date examDate) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        LinkedMultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        addFilesToBody(body, "student_files", studentFiles);
        addFilesToBody(body, "answer_files", answerFiles);
        body.add("exam_date", examDate);

        HttpEntity<LinkedMultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
        ResponseEntity<byte[]> responseEntity = restTemplate.exchange(
                AppConstants.FLASK_SERVER_URL, HttpMethod.POST, requestEntity, byte[].class);

        return responseEntity.getBody();
    }

    private void addFilesToBody(LinkedMultiValueMap<String, Object> body, String key, MultipartFile[] files) {
        for (MultipartFile file : files) {
            try {
                body.add(key, new ByteArrayResource(file.getBytes()) {
                    @Override
                    public String getFilename() {
                        return file.getOriginalFilename();
                    }
                });
            } catch (IOException e) {
                log.error("Error reading file: {}", file.getOriginalFilename(), e);
                throw new RuntimeException("Error reading file: " + file.getOriginalFilename(), e);
            }
        }
    }

    private void processZipFile(byte[] zipFileBytes, Date examDate, Timestamp submitDate, String userEmail) 
            throws IOException {
        
        Long savedExcelFileId = extractAndSaveExcelFile(zipFileBytes, examDate, submitDate, userEmail);
        
        if (savedExcelFileId == null) {
            throw new IllegalStateException("Excel 파일이 존재하지 않습니다.");
        }

        extractAndSaveImageFiles(zipFileBytes, savedExcelFileId, examDate, submitDate, userEmail);
    }

    private Long extractAndSaveExcelFile(byte[] zipFileBytes, Date examDate, Timestamp submitDate, String userEmail) 
            throws IOException {
        
        try (ZipInputStream zipInputStream = new ZipInputStream(new ByteArrayInputStream(zipFileBytes))) {
            ZipEntry entry;
            while ((entry = zipInputStream.getNextEntry()) != null) {
                if (entry.getName().endsWith(AppConstants.EXCEL_FILE_EXTENSION)) {
                    byte[] fileData = readZipEntry(zipInputStream);
                    
                    ExcelFile excelFile = new ExcelFile();
                    excelFile.setFileName(entry.getName());
                    excelFile.setData(fileData);
                    excelFile.setExamDate(examDate);
                    excelFile.setSubmitDate(submitDate);
                    excelFile.setEmail(userEmail);
                    
                    ExcelFile savedExcelFile = excelFileRepository.save(excelFile);
                    return savedExcelFile.getId();
                }
                zipInputStream.closeEntry();
            }
        }
        return null;
    }

    private void extractAndSaveImageFiles(byte[] zipFileBytes, Long excelFileId, Date examDate, 
                                         Timestamp submitDate, String userEmail) throws IOException {
        
        try (ZipInputStream zipInputStream = new ZipInputStream(new ByteArrayInputStream(zipFileBytes))) {
            ZipEntry entry;
            while ((entry = zipInputStream.getNextEntry()) != null) {
                if (!entry.getName().endsWith(AppConstants.EXCEL_FILE_EXTENSION)) {
                    byte[] imageData = readZipEntry(zipInputStream);
                    
                    ImageFile imageFile = new ImageFile();
                    imageFile.setExcelId(excelFileId);
                    imageFile.setImageName(entry.getName());
                    imageFile.setData(imageData);
                    imageFile.setExamDate(examDate);
                    imageFile.setSubmitDate(submitDate);
                    imageFile.setEmail(userEmail);
                    
                    imageFileRepository.save(imageFile);
                }
                zipInputStream.closeEntry();
            }
        }
    }

    private byte[] readZipEntry(ZipInputStream zipInputStream) throws IOException {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        byte[] buffer = new byte[AppConstants.BUFFER_SIZE];
        int len;
        while ((len = zipInputStream.read(buffer)) > -1) {
            outputStream.write(buffer, 0, len);
        }
        return outputStream.toByteArray();
    }
}

