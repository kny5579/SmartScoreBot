package org.example.smartScore.Controller;

import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.service.FileService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/files") // 클래스 레벨의 경로 설정
public class FileController {

    @Autowired
    private FileService fileService;

    @PostMapping("/by-date") // 메소드 레벨의 경로 설정
    public Map<String, List<?>> getFilesByDate(@RequestBody Map<String, String> request) {
        String date = request.get("date");
        List<ExcelFile> excelFiles = fileService.getExcelFilesByDate(date);
        List<ImageFile> images = fileService.getImagesByDate(date);

        Map<String, List<?>> response = new HashMap<>();
        response.put("excelFiles", excelFiles);
        response.put("images", images);

        return response;
    }
}
