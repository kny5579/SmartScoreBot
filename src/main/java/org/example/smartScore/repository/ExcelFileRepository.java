package org.example.smartScore.repository;

import org.example.smartScore.domain.ExcelFile;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.Date;
import java.util.List;
import java.util.Optional;

public interface ExcelFileRepository extends JpaRepository<ExcelFile, Long> {

    @Query("SELECT e FROM ExcelFile e WHERE e.submitDate = (SELECT MAX(e2.submitDate) FROM ExcelFile e2)")
    ExcelFile findLatestSubmitDateExcelFile();
    List<ExcelFile> findByExamDate(Date examDate);
}