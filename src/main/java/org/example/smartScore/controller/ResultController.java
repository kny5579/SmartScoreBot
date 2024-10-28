package org.example.smartScore.controller;

import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.repository.ExcelFileRepository;
import org.example.smartScore.repository.ImageFileRepository;
import org.example.smartScore.repository.StudentGradesRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

@Controller
public class ResultController {

    @Autowired
    private ExcelFileRepository excelFileRepository;

    @Autowired
    private ImageFileRepository imageFileRepository;

    @Autowired
    private StudentGradesRepository studentGradesRepository;

    @GetMapping("/result")
    public String showResultPage() {
        return "result";
    }

    @GetMapping("/resultData")
    public String getResultData(@RequestParam("exam_Date") String dateString, Model model) {
        try {
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
            Date date = dateFormat.parse(dateString);

            List<ImageFile> imageFiles = imageFileRepository.findByDate(date);
            List<ExcelFile> excelFiles = excelFileRepository.findByDate(date);

            model.addAttribute("examDate", dateString);
            model.addAttribute("imageFiles", imageFiles);
            model.addAttribute("excelFiles", excelFiles);
        } catch (Exception e) {
            e.printStackTrace();
            // 예외 처리 로직 추가
        }
        return "result";
    }

    @GetMapping("/download/excel")
    public ResponseEntity<InputStreamResource> downloadExcel(@RequestParam("download_date") String dateString) {
        try {
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
            Date date = dateFormat.parse(dateString);

            List<ExcelFile> excelFiles = excelFileRepository.findByDate(date);
            if (excelFiles.isEmpty()) {
                throw new RuntimeException("No files found for the specified date");
            }

            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            ZipOutputStream zipOutputStream = new ZipOutputStream(byteArrayOutputStream);

            Map<String, Integer> fileNameCountMap = new HashMap<>();
            for (ExcelFile excelFile : excelFiles) {
                String baseFileName = excelFile.getFileName();
                String fileName = baseFileName;
                int count = 1;

                while (fileNameCountMap.containsKey(fileName)) {
                    fileName = baseFileName.replaceFirst("(\\.[^.]+)$", "_" + count + "$1");
                    count++;
                }

                fileNameCountMap.put(fileName, count);

                ZipEntry zipEntry = new ZipEntry(fileName);
                zipEntry.setSize(excelFile.getData().length);
                zipOutputStream.putNextEntry(zipEntry);
                zipOutputStream.write(excelFile.getData());
                zipOutputStream.closeEntry();
            }

            zipOutputStream.flush(); // Ensure all data is written to the output stream
            zipOutputStream.close();

            ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());
            InputStreamResource resource = new InputStreamResource(byteArrayInputStream);

            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment;filename=excel_files.zip")
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .contentLength(byteArrayOutputStream.size())
                    .body(resource);
        } catch (Exception e) {
            e.printStackTrace();
            // 예외 처리 로직 추가
        }
        return ResponseEntity.badRequest().build();
    }

    @GetMapping("/scoreDistribution")
    @ResponseBody
    public List<Integer> getScoreDistribution(@RequestParam("exam_Date") String dateString) {
        try {
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
            Date date = dateFormat.parse(dateString);
            return studentGradesRepository.findScoresByDate(date); // 날짜에 따른 점수 데이터 반환
        } catch (Exception e) {
            e.printStackTrace();
            return Collections.emptyList(); // 오류 발생 시 빈 리스트 반환
        }
    }
}
