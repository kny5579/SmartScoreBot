package org.example.smartScore.controller;

import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.repository.ExcelFileRepository;
import org.example.smartScore.repository.ImageFileRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.ui.Model;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Optional;

@Controller
public class ViewController {

    @Autowired
    private ExcelFileRepository excelFileRepository;

    @Autowired
    private ImageFileRepository imageFileRepository;

    @GetMapping("/download/excel") //엑셀 다운로드
    public ResponseEntity<ByteArrayResource> downloadExcelFile(@RequestParam("download_date") String downloadDateString) throws ParseException {
        // 날짜 문자열을 Date 객체로 변환
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
        Date downloadDate = dateFormat.parse(downloadDateString);

        Optional<ExcelFile> optionalFile = excelFileRepository.findByDateOne(downloadDate);

        if (!optionalFile.isPresent()) {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }

        ExcelFile file = optionalFile.get();
        ByteArrayResource resource = new ByteArrayResource(file.getData());

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment;filename=" + file.getFileName())
                .contentType(org.springframework.http.MediaType.APPLICATION_OCTET_STREAM)
                .body(resource);
    }

    @GetMapping("/result") //채점된 이미지 파일 저장
    public String showResult(@RequestParam("exam_date") String examDateString, Model model) throws Exception {
        // 날짜 문자열을 Date 객체로 변환
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
        Date examDate = dateFormat.parse(examDateString);

        // 해당 날짜의 이미지 파일 목록 가져오기
        List<ImageFile> files = new ArrayList<>();
        imageFileRepository.findByDate(examDate);

        if (files.isEmpty()) {
            return "error";
        }

        // 모델에 파일 목록 추가
        model.addAttribute("files", files);
        model.addAttribute("examDate", examDateString);

        return "result";
    }
}