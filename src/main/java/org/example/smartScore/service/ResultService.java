package org.example.smartScore.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.constants.AppConstants;
import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.repository.ExcelFileRepository;
import org.example.smartScore.repository.ImageFileRepository;
import org.example.smartScore.repository.StudentGradesRepository;
import org.example.smartScore.util.DateUtils;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.text.ParseException;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.*;

import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class ResultService {

    private final ExcelFileRepository excelFileRepository;
    private final ImageFileRepository imageFileRepository;
    private final StudentGradesRepository studentGradesRepository;

    public ResultData getLatestResultData(String userEmail) {
        ExcelFile latestExcelFile = excelFileRepository.findLatestSubmitDateExcelFileByEmail(userEmail);
        
        if (latestExcelFile == null) {
            return null;
        }

        Date examDate = latestExcelFile.getExamDate();
        String dateString = DateUtils.formatDate(examDate);

        List<ImageFile> imageFiles = imageFileRepository.findByExamDateAndEmail(examDate, userEmail);
        List<ExcelFile> excelFiles = excelFileRepository.findByExamDateAndEmail(examDate, userEmail);

        return new ResultData(dateString, imageFiles, excelFiles);
    }

    public ResultData getResultDataByDate(String dateString, String userEmail) throws ParseException {
        Date date = DateUtils.parseDate(dateString);
        
        List<ImageFile> imageFiles = imageFileRepository.findByExamDateAndEmail(date, userEmail);
        List<ExcelFile> excelFiles = excelFileRepository.findByExamDateAndEmail(date, userEmail);

        return new ResultData(dateString, imageFiles, excelFiles);
    }

    public ResultDetailData getResultDetail(Long id) {
        List<ImageFile> imageFiles = imageFileRepository.findByExcelId(id);
        ExcelFile excelFile = excelFileRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Excel file not found with id: " + id));

        Date date = excelFile.getExamDate();
        LocalDate examDate = date.toInstant()
                .atZone(ZoneId.systemDefault())
                .toLocalDate();
        return new ResultDetailData(examDate, imageFiles, excelFile);
    }

    public ResponseEntity<InputStreamResource> downloadExcelFiles(String dateString, String userEmail) 
            throws ParseException {
        
        Date date = DateUtils.parseDate(dateString);
        List<ExcelFile> excelFiles = excelFileRepository.findByExamDateAndEmail(date, userEmail);
        
        if (excelFiles.isEmpty()) {
            throw new IllegalArgumentException("No files found for the specified date");
        }

        byte[] zipData = createZipFromExcelFiles(excelFiles);
        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(zipData);
        InputStreamResource resource = new InputStreamResource(byteArrayInputStream);

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment;filename=" + AppConstants.ZIP_FILE_NAME)
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .contentLength(zipData.length)
                .body(resource);
    }

    public List<Integer> getScoreDistribution(String dateString, String userEmail) throws ParseException {
        LocalDate date = LocalDate.parse(dateString);

        LocalDateTime startLdt = date.atStartOfDay();
        LocalDateTime endLdt = date.plusDays(1).atStartOfDay();

        Date start = Date.from(startLdt.atZone(ZoneId.systemDefault()).toInstant());
        Date end = Date.from(endLdt.atZone(ZoneId.systemDefault()).toInstant());

        return studentGradesRepository.findScoresByDate(start, end, userEmail);
    }

    @Transactional
    public void deleteRecord(Long id) {
        imageFileRepository.deleteByExcelId(id);
        excelFileRepository.deleteById(id);
        log.info("Record deleted successfully: id={}", id);
    }

    private byte[] createZipFromExcelFiles(List<ExcelFile> excelFiles) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        
        try (ZipOutputStream zipOutputStream = new ZipOutputStream(byteArrayOutputStream)) {
            Map<String, Integer> fileNameCountMap = new HashMap<>();
            
            for (ExcelFile excelFile : excelFiles) {
                String fileName = generateUniqueFileName(excelFile.getFileName(), fileNameCountMap);
                fileNameCountMap.put(fileName, fileNameCountMap.getOrDefault(fileName, 0) + 1);

                ZipEntry zipEntry = new ZipEntry(fileName);
                zipEntry.setSize(excelFile.getData().length);
                zipOutputStream.putNextEntry(zipEntry);
                zipOutputStream.write(excelFile.getData());
                zipOutputStream.closeEntry();
            }
        } catch (Exception e) {
            log.error("Error creating zip file", e);
            throw new RuntimeException("Error creating zip file", e);
        }
        
        return byteArrayOutputStream.toByteArray();
    }

    private String generateUniqueFileName(String baseFileName, Map<String, Integer> fileNameCountMap) {
        String fileName = baseFileName;
        int count = 1;

        while (fileNameCountMap.containsKey(fileName)) {
            fileName = baseFileName.replaceFirst("(\\.[^.]+)$", "_" + count + "$1");
            count++;
        }

        return fileName;
    }

    public record ResultData(String examDate, List<ImageFile> imageFiles, List<ExcelFile> excelFiles) {}
    public record ResultDetailData(LocalDate examDate, List<ImageFile> imageFiles, ExcelFile excelFile) {}
}

