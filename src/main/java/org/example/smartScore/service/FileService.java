package org.example.smartScore.service;

import lombok.RequiredArgsConstructor;
import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.repository.ExcelFileRepository;
import org.example.smartScore.repository.ImageFileRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class FileService {

    private final ExcelFileRepository excelFileRepository;
    private final ImageFileRepository imageFileRepository;

    public List<ExcelFile> getExcelFilesByDate(String date) {
        return excelFileRepository.findByDate(date);
    }

    public List<ImageFile> getImagesByDate(String date) {
        return imageFileRepository.findByDate(date);
    }
}