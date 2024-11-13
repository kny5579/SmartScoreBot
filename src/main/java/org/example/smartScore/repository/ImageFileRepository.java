package org.example.smartScore.repository;

import org.example.smartScore.domain.ImageFile;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Date;
import java.util.List;
import java.util.Optional;

public interface ImageFileRepository extends JpaRepository<ImageFile, Long> {
    List<ImageFile> findBySubmitDate(Date submitDate);
    List<ImageFile> findByExamDate(Date examDate);
    Optional<ImageFile> findById(Long id);
}