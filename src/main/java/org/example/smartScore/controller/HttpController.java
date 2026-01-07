package org.example.smartScore.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.service.FileUploadService;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.security.Principal;
import java.text.ParseException;

@Slf4j
@Controller
@RequiredArgsConstructor
public class HttpController {

    private final FileUploadService fileUploadService;

    @PostMapping("/upload")
    public ResponseEntity<String> uploadImages(
            @RequestParam("student_files") MultipartFile[] studentFiles,
            @RequestParam("answer_files") MultipartFile[] answerFiles,
            @RequestParam("exam_date") String dateString,
            Principal principal) throws IOException, ParseException {

        String userEmail = principal.getName();
        log.info("File upload request received from user: {}, exam date: {}", userEmail, dateString);

        String result = fileUploadService.uploadAndProcessFiles(studentFiles, answerFiles, dateString, userEmail);
        return ResponseEntity.ok(result);
    }
}
