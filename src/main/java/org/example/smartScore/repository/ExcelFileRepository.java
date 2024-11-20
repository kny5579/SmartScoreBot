package org.example.smartScore.repository;

import org.example.smartScore.domain.ExcelFile;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.Date;
import java.util.List;
import java.util.Optional;

public interface ExcelFileRepository extends JpaRepository<ExcelFile, Long> {

    @Query("SELECT e FROM ExcelFile e WHERE e.email = :email ORDER BY e.submitDate DESC LIMIT 1")
    ExcelFile findLatestSubmitDateExcelFileByEmail(@Param("email") String email);

    @Query("SELECT e FROM ExcelFile e WHERE e.examDate = :examDate AND e.email = :email")
    List<ExcelFile> findByExamDateAndEmail(@Param("examDate") Date examDate, @Param("email") String email);

    List<ExcelFile> findByExamDate(Date examDate);

    List<ExcelFile> findByEmail(String email);
}