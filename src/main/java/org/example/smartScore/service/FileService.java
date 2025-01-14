package org.example.smartScore.service;

import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.repository.ExcelFileRepository;
import org.example.smartScore.repository.ImageFileRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class FileService {

    @Autowired
    private ExcelFileRepository excelFileRepository;

    @Autowired
    private ImageFileRepository imageFileRepository;

    public List<ExcelFile> getExcelFilesByDate(String date) {
        return excelFileRepository.findByDate(date);
    }

    public List<ImageFile> getImagesByDate(String date) {
        return imageFileRepository.findByDate(date);
    }
}