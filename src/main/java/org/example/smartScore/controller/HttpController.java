package org.example.smartScore.controller;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Controller
public class HttpController {

    // 이미지 업로드 및 날짜 데이터를 Flask 서버로 보냄
    @PostMapping("/upload")
    public String uploadImages(@RequestParam("file") MultipartFile[] files, @RequestParam("date") String date, Model model) throws IOException {

        // Flask 서버의 URL
        String flaskUrl = "http://localhost:5000/upload";

        // 헤더 타입 설정
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        // 이미지 파일과 날짜 데이터를 MultiValueMap에 추가
        LinkedMultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        for (MultipartFile file : files) { //이미지 파일 추가
            body.add("file", new ByteArrayResource(file.getBytes()) { //파일을 byteArray로 변환해서 맵에 추가
                @Override
                public String getFilename() { //파일 이름 유지 위한 오버라이딩
                    return file.getOriginalFilename();
                }
            });
        }
        body.add("date", date); //날짜 추가

        // HTTP 요청 생성
        HttpEntity<LinkedMultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        // RestTemplate을 사용하여 Flask 서버에 POST 요청 전송
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<byte[]> responseEntity = restTemplate.exchange(flaskUrl, HttpMethod.POST, requestEntity, byte[].class);

        // Flask 서버로부터 받은 응답 처리
        byte[] excelFileBytes = responseEntity.getBody();

        // 모델에 응답 결과 추가
        model.addAttribute("excelFileBytes", excelFileBytes);

        return "result";
    }

    // 파일 다운로드 링크 컨트롤러
    @GetMapping("/download")
    public ResponseEntity<ByteArrayResource> downloadExcelFile(@ModelAttribute("excelFileBytes") byte[] excelFileBytes) {
        ByteArrayResource resource = new ByteArrayResource(excelFileBytes);
        HttpHeaders headers = new HttpHeaders();
        headers.add(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=excel_file.xlsx");
        return ResponseEntity.ok()
                .headers(headers)
                .contentLength(excelFileBytes.length)
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .body(resource);
    }
}
