package org.example.smartScore.repository;

import org.example.smartScore.domain.ImageFile;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigInteger;
import java.util.Date;
import java.util.List;
import java.util.Optional;

public interface ImageFileRepository extends JpaRepository<ImageFile, Long> {
    @Query("SELECT i FROM ImageFile i WHERE i.examDate = :examDate AND i.email = :email")
    List<ImageFile> findByExamDateAndEmail(@Param("examDate") Date examDate, @Param("email") String email);
    List<ImageFile> findByExamDate(Date examDate);
    List<ImageFile> findByExcelId(Long excelId);
    Optional<ImageFile> findById(Long id);

    @Transactional
    void deleteByExcelId(Long excelId);
}