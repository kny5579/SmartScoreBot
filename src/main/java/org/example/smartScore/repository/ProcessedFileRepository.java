package org.example.smartScore.repository;

import org.example.smartScore.domain.ProcessedFileEntity;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ProcessedFileRepository extends JpaRepository<ProcessedFileEntity, Long> {
}
