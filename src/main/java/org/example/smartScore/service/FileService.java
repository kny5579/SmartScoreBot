package org.example.smartScore.service;

import lombok.RequiredArgsConstructor;
import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.repository.ExcelFileRepository;
import org.example.smartScore.repository.ImageFileRepository;
import org.example.smartScore.util.DateUtils;
import org.springframework.stereotype.Service;

import java.text.ParseException;
import java.util.Collections;
import java.util.List;

@Service
@RequiredArgsConstructor
public class FileService {

    private final ExcelFileRepository excelFileRepository;
    private final ImageFileRepository imageFileRepository;

    public List<ExcelFile> getExcelFilesByDate(String date) {
        try {
            return excelFileRepository.findByExamDate(DateUtils.parseDate(date));
        } catch (ParseException e) {
            return Collections.emptyList();
        }
    }

    public List<ImageFile> getImagesByDate(String date) {
        try {
            return imageFileRepository.findByExamDate(DateUtils.parseDate(date));
        } catch (ParseException e) {
            return Collections.emptyList();
        }
    }
}