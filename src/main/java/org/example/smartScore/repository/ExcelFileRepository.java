package org.example.smartScore.repository;

import org.example.smartScore.domain.ExcelFile;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.Date;
import java.util.List;
import java.util.Optional;

public interface ExcelFileRepository extends JpaRepository<ExcelFile, Long> {
    List<ExcelFile> findByDate(Date date);

    @Query("SELECT MAX(e.date) FROM excel_file e")
    Optional<Date> findLatestDate();
}