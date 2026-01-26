package org.example.smartScore.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.service.ResultService;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.security.Principal;
import java.text.ParseException;
import java.util.Collections;
import java.util.List;

@Slf4j
@Controller
@RequiredArgsConstructor
public class ResultController {

    private final ResultService resultService;

    @GetMapping("/resultDate")
    public String showResultDatePage() {
        return "resultDate";
    }

    @GetMapping("/result")
    public String getResultData(Model model, Principal principal) {
        try {
            String userEmail = principal.getName();
            ResultService.ResultData resultData = resultService.getLatestResultData(userEmail);

            if (resultData != null) {
                model.addAttribute("examDate", resultData.examDate());
                model.addAttribute("imageFiles", resultData.imageFiles());
                model.addAttribute("excelFiles", resultData.excelFiles());
            } else {
                // 데이터가 없을 때도 빈 문자열로 설정하여 JavaScript 오류 방지
                model.addAttribute("examDate", "");
                model.addAttribute("imageFiles", Collections.emptyList());
                model.addAttribute("excelFiles", Collections.emptyList());
            }
        } catch (Exception e) {
            log.error("Error fetching result data", e);
            model.addAttribute("examDate", "");
            model.addAttribute("imageFiles", Collections.emptyList());
            model.addAttribute("excelFiles", Collections.emptyList());
        }
        return "result";
    }

    @GetMapping("/resultData")
    public String getResultDataByDate(@RequestParam("exam_Date") String dateString, Model model, Principal principal) {
        try {
            String userEmail = principal.getName();
            ResultService.ResultData resultData = resultService.getResultDataByDate(dateString, userEmail);

            model.addAttribute("examDate", resultData.examDate());
            model.addAttribute("imageFiles", resultData.imageFiles());
            model.addAttribute("excelFiles", resultData.excelFiles());
        } catch (ParseException e) {
            log.error("Error parsing date: {}", dateString, e);
            model.addAttribute("examDate", dateString); // 입력된 날짜를 그대로 전달
            model.addAttribute("imageFiles", Collections.emptyList());
            model.addAttribute("excelFiles", Collections.emptyList());
        } catch (Exception e) {
            log.error("Error fetching result data for date: {}", dateString, e);
            model.addAttribute("examDate", dateString); // 입력된 날짜를 그대로 전달
            model.addAttribute("imageFiles", Collections.emptyList());
            model.addAttribute("excelFiles", Collections.emptyList());
        }
        return "resultDate";
    }

    @GetMapping("/resultDetail/{id}")
    public String getResultDetail(@PathVariable Long id, Model model) {
        try {
            ResultService.ResultDetailData resultDetail = resultService.getResultDetail(id);

            model.addAttribute("examDate", resultDetail.examDate());
            model.addAttribute("imageFiles", resultDetail.imageFiles());
            model.addAttribute("excelFiles", resultDetail.excelFile());
        } catch (Exception e) {
            log.error("Error fetching result detail for id: {}", id, e);
        }
        return "resultDetail";
    }

    @GetMapping("/download/excel")
    public ResponseEntity<InputStreamResource> downloadExcel(
            @RequestParam("download_date") String dateString, Principal principal) {
        try {
            String userEmail = principal.getName();
            return resultService.downloadExcelFiles(dateString, userEmail);
        } catch (ParseException e) {
            log.error("Error parsing date: {}", dateString, e);
            return ResponseEntity.badRequest().build();
        } catch (Exception e) {
            log.error("Error downloading excel files for date: {}", dateString, e);
            return ResponseEntity.badRequest().build();
        }
    }

    @GetMapping("/scoreDistribution")
    @ResponseBody
    public List<Integer> getScoreDistribution(@RequestParam("exam_Date") String dateString, Principal principal) {
        try {
            String userEmail = principal.getName();
            return resultService.getScoreDistribution(dateString, userEmail);
        } catch (ParseException e) {
            log.error("Error parsing date: {}", dateString, e);
            return Collections.emptyList();
        } catch (Exception e) {
            log.error("Error fetching score distribution for date: {}", dateString, e);
            return Collections.emptyList();
        }
    }

    @DeleteMapping("/api/delete/{id}")
    public ResponseEntity<String> deleteRecord(@PathVariable Long id) {
        try {
            resultService.deleteRecord(id);
            return ResponseEntity.ok("삭제되었습니다.");
        } catch (Exception e) {
            log.error("Error deleting record with id: {}", id, e);
            return ResponseEntity.badRequest().body("삭제에 실패했습니다.");
        }
    }
}
