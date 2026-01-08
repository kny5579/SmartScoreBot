package org.example.smartScore.service;

import lombok.extern.slf4j.Slf4j;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.example.smartScore.service.GradingService.GradingResult;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
public class ExcelGenerationService {

    private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd");

    public byte[] generateGradingExcel(List<GradingResult> gradingResults, Date examDate) throws IOException {
        try (Workbook workbook = new XSSFWorkbook()) {
            Sheet sheet = workbook.createSheet("채점 결과");

            // 스타일 생성
            CellStyle headerStyle = createHeaderStyle(workbook);
            CellStyle scoreStyle = createScoreStyle(workbook);
            CellStyle correctStyle = createCorrectStyle(workbook);
            CellStyle incorrectStyle = createIncorrectStyle(workbook);

            // 헤더 생성
            int rowNum = 0;
            Row headerRow = sheet.createRow(rowNum++);
            headerRow.createCell(0).setCellValue("학번");
            headerRow.createCell(1).setCellValue("점수");
            headerRow.createCell(2).setCellValue("정답 수");
            headerRow.createCell(3).setCellValue("총 문제 수");
            headerRow.createCell(4).setCellValue("정답률");
            headerRow.createCell(5).setCellValue("시험 날짜");

            // 헤더 스타일 적용
            for (int i = 0; i < 6; i++) {
                headerRow.getCell(i).setCellStyle(headerStyle);
            }

            // 데이터 입력
            for (GradingResult result : gradingResults) {
                Row row = sheet.createRow(rowNum++);
                
                row.createCell(0).setCellValue(result.studentId());
                row.createCell(1).setCellValue(result.score());
                row.createCell(2).setCellValue(result.correctCount());
                row.createCell(3).setCellValue(result.totalQuestions());
                
                double accuracy = result.totalQuestions() > 0 ?
                        (double) result.correctCount() / result.totalQuestions() * 100 : 0;
                row.createCell(4).setCellValue(String.format("%.2f%%", accuracy));
                row.createCell(5).setCellValue(DATE_FORMAT.format(examDate));

                // 점수 셀에 스타일 적용
                row.getCell(1).setCellStyle(scoreStyle);
            }

            // 컬럼 너비 자동 조정
            for (int i = 0; i < 6; i++) {
                sheet.autoSizeColumn(i);
            }

            // Excel 파일을 byte 배열로 변환
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            workbook.write(outputStream);
            return outputStream.toByteArray();
        }
    }

    private CellStyle createHeaderStyle(Workbook workbook) {
        CellStyle style = workbook.createCellStyle();
        Font font = workbook.createFont();
        font.setBold(true);
        font.setFontHeightInPoints((short) 12);
        style.setFont(font);
        style.setFillForegroundColor(IndexedColors.GREY_25_PERCENT.getIndex());
        style.setFillPattern(FillPatternType.SOLID_FOREGROUND);
        style.setBorderBottom(BorderStyle.THIN);
        style.setBorderTop(BorderStyle.THIN);
        style.setBorderLeft(BorderStyle.THIN);
        style.setBorderRight(BorderStyle.THIN);
        return style;
    }

    private CellStyle createScoreStyle(Workbook workbook) {
        CellStyle style = workbook.createCellStyle();
        Font font = workbook.createFont();
        font.setBold(true);
        font.setColor(IndexedColors.BLUE.getIndex());
        style.setFont(font);
        return style;
    }

    private CellStyle createCorrectStyle(Workbook workbook) {
        CellStyle style = workbook.createCellStyle();
        style.setFillForegroundColor(IndexedColors.LIGHT_GREEN.getIndex());
        style.setFillPattern(FillPatternType.SOLID_FOREGROUND);
        return style;
    }

    private CellStyle createIncorrectStyle(Workbook workbook) {
        CellStyle style = workbook.createCellStyle();
        style.setFillForegroundColor(IndexedColors.CORAL.getIndex());
        style.setFillPattern(FillPatternType.SOLID_FOREGROUND);
        return style;
    }
}

