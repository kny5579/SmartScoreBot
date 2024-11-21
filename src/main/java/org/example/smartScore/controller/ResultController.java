package org.example.smartScore.controller;

import jakarta.transaction.Transactional;
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
import org.springframework.web.bind.annotation.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.security.Principal;
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

    @GetMapping("/resultDate")
    public String showResultDatePage() {
        return "resultDate";
    }

    @GetMapping("/result")
    public String getResultData(Model model, Principal principal) {
        try {
            // 현재 로그인된 유저의 이메일 가져오기
            String userEmail = principal.getName();

            // 최신 제출 날짜의 Excel 파일 가져오기 (유저 기준)
            ExcelFile latestExcelFile = excelFileRepository.findLatestSubmitDateExcelFileByEmail(userEmail);

            if (latestExcelFile != null) {
                Date date = latestExcelFile.getExamDate();
                SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
                String dateString = dateFormat.format(date);

                // 해당 유저와 시험 날짜 기준으로 이미지 및 Excel 파일 가져오기
                List<ImageFile> imageFiles = imageFileRepository.findByExamDateAndEmail(date, userEmail);
                List<ExcelFile> excelFiles = excelFileRepository.findByExamDateAndEmail(date, userEmail);

                model.addAttribute("examDate", dateString);
                model.addAttribute("imageFiles", imageFiles);
                model.addAttribute("excelFiles", excelFiles);
            }
        } catch (Exception e) {
            System.err.println("Error fetching result data: " + e.getMessage());
        }
        return "result";
    }


    @GetMapping("/resultData")
    public String getResultData(@RequestParam("exam_Date") String dateString, Model model, Principal principal) {
        try {
            String userEmail = principal.getName();
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
            Date date = dateFormat.parse(dateString);

            List<ImageFile> imageFiles = imageFileRepository.findByExamDateAndEmail(date, userEmail);
            List<ExcelFile> excelFiles = excelFileRepository.findByExamDateAndEmail(date, userEmail);

            model.addAttribute("examDate", dateString);
            model.addAttribute("imageFiles", imageFiles);
            model.addAttribute("excelFiles", excelFiles);
        } catch (Exception e) {
            System.err.println("Error fetching result data for date " + dateString + ": " + e.getMessage());
        }
        return "resultDate";
    }

    @GetMapping("/resultDetail/{id}")
    public String getResultDetail(@PathVariable Long id, Model model) {
        try {
            List<ImageFile> imageFiles = imageFileRepository.findByExcelId(id);
            Optional<ExcelFile> excelFiles = excelFileRepository.findById(id);

            model.addAttribute("examDate", excelFiles.get().getExamDate());
            model.addAttribute("imageFiles", imageFiles);
            model.addAttribute("excelFiles", excelFiles);
        } catch (Exception e) {
            System.err.println("Error fetching result detail: " + e.getMessage());
        }
        return "resultDetail";
    }

    // Excel 파일 다운로드
    @GetMapping("/download/excel")
    public ResponseEntity<InputStreamResource> downloadExcel(@RequestParam("download_date") String dateString, Principal principal) {
        try {
            String userEmail = principal.getName();
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
            Date date = dateFormat.parse(dateString);

            List<ExcelFile> excelFiles = excelFileRepository.findByExamDateAndEmail(date, userEmail);
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

            zipOutputStream.close();

            ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());
            InputStreamResource resource = new InputStreamResource(byteArrayInputStream);

            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment;filename=excel_files.zip")
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .contentLength(byteArrayOutputStream.size())
                    .body(resource);
        } catch (Exception e) {
            System.err.println("Error downloading excel files for date " + dateString + ": " + e.getMessage());
        }
        return ResponseEntity.badRequest().build();
    }

    @GetMapping("/scoreDistribution")
    @ResponseBody
    public List<Integer> getScoreDistribution(@RequestParam("exam_Date") String dateString, Principal principal) {
        try {
            String userEmail = principal.getName();
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
            Date date = dateFormat.parse(dateString);

            List<Integer> scores = studentGradesRepository.findByExamDateAndEmail(date, userEmail);
            System.out.println("Fetched scores for chart: " + scores);
            return scores;
        } catch (Exception e) {
            System.err.println("Error fetching score distribution for date " + dateString + ": " + e.getMessage());
            return Collections.emptyList();
        }
    }

    @Transactional
    @DeleteMapping("/api/delete/{id}")
    public ResponseEntity<String> deleteRecord(@PathVariable Long id){
        imageFileRepository.deleteByExcelId(id);
        excelFileRepository.deleteById(id);
        return ResponseEntity.ok("삭제되었습니다.");
    }
}
