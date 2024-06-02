package org.example.smartScore.controller;

import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.repository.ExcelFileRepository;
import org.example.smartScore.repository.ImageFileRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Optional;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

@Controller
public class ResultController {

    @Autowired
    private ExcelFileRepository excelFileRepository;

    @Autowired
    private ImageFileRepository imageFileRepository;

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
            Optional<ExcelFile> excelFile = excelFileRepository.findByDate(date);

            model.addAttribute("examDate", dateString);
            model.addAttribute("imageFiles", imageFiles);
            model.addAttribute("excelFile", excelFile.orElse(null));
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

            Optional<ExcelFile> excelFileOptional = excelFileRepository.findByDate(date);
            if (excelFileOptional.isEmpty()) {
                throw new RuntimeException("No files found for the specified date");
            }
            ExcelFile excelFile = excelFileOptional.get();

            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            ZipOutputStream zipOutputStream = new ZipOutputStream(byteArrayOutputStream);

            ZipEntry zipEntry = new ZipEntry(excelFile.getFileName());
            zipEntry.setSize(excelFile.getData().length);
            zipOutputStream.putNextEntry(zipEntry);
            zipOutputStream.write(excelFile.getData());
            zipOutputStream.closeEntry();

            zipOutputStream.close();

            ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());
            InputStreamResource resource = new InputStreamResource(byteArrayInputStream);

            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment;filename=excel_file.zip")
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .contentLength(byteArrayOutputStream.size())
                    .body(resource);
        } catch (Exception e) {
            e.printStackTrace();
            // 예외 처리 로직 추가
        }
        return ResponseEntity.badRequest().build();
    }
}
