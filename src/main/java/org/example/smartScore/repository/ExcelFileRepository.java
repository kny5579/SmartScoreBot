package org.example.smartScore.repository;

import org.example.smartScore.domain.ExcelFile;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Date;
import java.util.List;
import java.util.Optional;

public interface ExcelFileRepository extends JpaRepository<ExcelFile, Long> {
    List<ExcelFile> findByDate(Date date);
}