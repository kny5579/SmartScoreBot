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
        String flaskUrl = "http://localhost:5000/upload";

        // 날짜 문자열을 Date 객체로 변환
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
        Date date = dateFormat.parse(dateString);

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
        body.add("date", date); // 날짜 추가

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

            // 각 파일 처리
            while ((entry = zipInputStream.getNextEntry()) != null) {
                String fileName = entry.getName();
                ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
                byte[] buffer = new byte[1024];
                int len;
                while ((len = zipInputStream.read(buffer)) > -1) {
                    outputStream.write(buffer, 0, len);
                }
                outputStream.close();

                // 파일의 확장자를 확인하여 각 엔티티에 저장
                if (fileName.endsWith(".xlsx")) {
                    // Excel 파일 저장
                    ExcelFile excelFile = new ExcelFile();
                    excelFile.setFileName(fileName);
                    excelFile.setData(outputStream.toByteArray());
                    excelFile.setDate(date);
                    excelFileRepository.save(excelFile);
                } else {
                    // Image 파일 저장
                    ImageFile imageFile = new ImageFile();
                    imageFile.setImageName(fileName);
                    imageFile.setData(outputStream.toByteArray());
                    imageFile.setDate(date);
                    imageFileRepository.save(imageFile);
                }

                zipInputStream.closeEntry();
            }
            zipInputStream.close();
        }

        return "redirect:/result";
    }
}
