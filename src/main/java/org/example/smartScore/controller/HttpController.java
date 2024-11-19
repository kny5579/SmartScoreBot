package org.example.smartScore.controller;

import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.repository.ExcelFileRepository;
import org.example.smartScore.repository.ImageFileRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Controller;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.math.BigInteger;
import java.sql.Timestamp; // Timestamp import 추가
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

@Controller
public class HttpController {

    @Autowired
    private ExcelFileRepository excelFileRepository;

    @Autowired
    private ImageFileRepository imageFileRepository;

    @PostMapping("/upload")
    public String uploadImages(@RequestParam("student_files") MultipartFile[] studentFiles,
                               @RequestParam("answer_files") MultipartFile[] answerFiles,
                               @RequestParam("exam_date") String dateString) throws IOException, ParseException {

        // Flask 서버의 URL
        String flaskUrl = "http://flaskserver:5000/upload";

        // 날짜 문자열을 Date 객체로 변환
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
        Date examDate = dateFormat.parse(dateString);
        System.out.println("Exam Date: " + examDate);

        Timestamp submitDate = new Timestamp(System.currentTimeMillis()); // 현재 시간을 포함하는 Timestamp 생성

        // 헤더 타입 설정
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        // 이미지 파일과 날짜 데이터를 MultiValueMap에 추가
        LinkedMultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        for (MultipartFile file : studentFiles) { // 학생 이미지 파일 추가
            body.add("student_files", new ByteArrayResource(file.getBytes()) { // 파일을 byteArray로 변환해서 맵에 추가
                @Override
                public String getFilename() { // 파일 이름 유지 위한 오버라이딩
                    return file.getOriginalFilename();
                }
            });
        }
        for (MultipartFile file : answerFiles) { // 정답 이미지 파일 추가
            body.add("answer_files", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            });
        }
        body.add("exam_date", examDate); // 시험 날짜 추가

        // HTTP 요청 생성
        HttpEntity<LinkedMultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        // RestTemplate 통해 Flask 서버에 POST 요청 전송
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<byte[]> responseEntity = restTemplate.exchange(flaskUrl, HttpMethod.POST, requestEntity, byte[].class);

        // Flask 서버 응답 처리(zip 파일)
        byte[] zipFileBytes = responseEntity.getBody();

        // ZIP 파일 해제
        if (zipFileBytes != null) {
            ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(zipFileBytes);
            ZipInputStream zipInputStream = new ZipInputStream(byteArrayInputStream);
            ZipEntry entry;
            Long savedExcelFileId = null; // ExcelFile ID 저장 변수

            // 1. 먼저 Excel 파일을 처리하여 ID를 저장
            while ((entry = zipInputStream.getNextEntry()) != null) {
                String fileName = entry.getName();
                ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
                byte[] buffer = new byte[1024];
                int len;
                while ((len = zipInputStream.read(buffer)) > -1) {
                    outputStream.write(buffer, 0, len);
                }
                outputStream.close();

                if (fileName.endsWith(".xlsx")) {
                    // Excel 파일 저장
                    ExcelFile excelFile = new ExcelFile();
                    excelFile.setFileName(fileName);
                    excelFile.setData(outputStream.toByteArray());
                    excelFile.setExamDate(examDate); // 시험 날짜 설정
                    excelFile.setSubmitDate(submitDate); // 제출 날짜 설정
                    ExcelFile savedExcelFile = excelFileRepository.save(excelFile);
                    savedExcelFileId = savedExcelFile.getId(); // ExcelFile ID 저장
                }
                zipInputStream.closeEntry();
            }

            // 2. Excel 파일 ID가 설정되었는지 확인
            if (savedExcelFileId == null) {
                throw new IllegalStateException("Excel 파일이 존재하지 않습니다.");
            }

            // 3. 다시 ZIP 스트림을 열어 이미지 파일 처리
            zipInputStream = new ZipInputStream(new ByteArrayInputStream(zipFileBytes));
            while ((entry = zipInputStream.getNextEntry()) != null) {
                String fileName = entry.getName();
                ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
                byte[] buffer = new byte[1024];
                int len;
                while ((len = zipInputStream.read(buffer)) > -1) {
                    outputStream.write(buffer, 0, len);
                }
                outputStream.close();

                if (!fileName.endsWith(".xlsx")) {
                    ImageFile imageFile = new ImageFile();
                    imageFile.setExcelId(savedExcelFileId);
                    imageFile.setImageName(fileName);
                    imageFile.setData(outputStream.toByteArray());
                    imageFile.setExamDate(examDate);
                    imageFile.setSubmitDate(submitDate);
                    imageFileRepository.save(imageFile);
                }
                zipInputStream.closeEntry();
            }
            zipInputStream.close();
        }

        return "redirect:/result";
    }
}
