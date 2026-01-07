package org.example.smartScore.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.domain.ProcessedFileEntity;
import org.example.smartScore.repository.ProcessedFileRepository;
import org.example.smartScore.util.DateUtils;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Controller
@RequiredArgsConstructor
public class ViewController {

    private final ProcessedFileRepository processedFileRepository;
    private static final String FILE_TYPE_EXCEL = "excel";
    private static final String FILE_TYPE_IMAGE = "image";

    @GetMapping("/download/excel")
    public ResponseEntity<ByteArrayResource> downloadExcelFile(
            @RequestParam("download_date") String downloadDateString) throws ParseException {
        
        return processedFileRepository.findByDateAndFileType(
                        DateUtils.parseDate(downloadDateString), FILE_TYPE_EXCEL)
                .map(file -> {
                    ByteArrayResource resource = new ByteArrayResource(file.getData());
                    return ResponseEntity.ok()
                            .header(HttpHeaders.CONTENT_DISPOSITION, "attachment;filename=" + file.getFilename())
                            .contentType(MediaType.APPLICATION_OCTET_STREAM)
                            .body(resource);
                })
                .orElse(new ResponseEntity<>(HttpStatus.NOT_FOUND));
    }

    @GetMapping("/result")
    public String showResult(@RequestParam("exam_date") String examDateString, Model model) {
        try {
            List<ProcessedFileEntity> files = new ArrayList<>();
            processedFileRepository.findByDateAndFileType(
                    DateUtils.parseDate(examDateString), FILE_TYPE_IMAGE)
                    .ifPresent(files::add);

            model.addAttribute("files", files);
            model.addAttribute("examDate", examDateString);
        } catch (ParseException e) {
            log.error("Error parsing exam date: {}", examDateString, e);
        }
        return "result";
    }
}
