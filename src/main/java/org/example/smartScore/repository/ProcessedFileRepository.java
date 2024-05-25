package org.example.smartScore.repository;

import org.example.smartScore.domain.ProcessedFileEntity;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Date;
import java.util.Optional;

public interface ProcessedFileRepository extends JpaRepository<ProcessedFileEntity, Long> {
    Optional<ProcessedFileEntity> findByDateAndFileType(Date date, String fileType);
}