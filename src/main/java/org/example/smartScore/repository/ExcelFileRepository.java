package org.example.smartScore.repository;

import org.example.smartScore.domain.ExcelFile;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ExcelFileRepository extends JpaRepository<ExcelFile, Long> {
    List<ExcelFile> findByDate(String date);
}