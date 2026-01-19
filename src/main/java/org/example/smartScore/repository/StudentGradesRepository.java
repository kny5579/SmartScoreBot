package org.example.smartScore.repository;

import org.example.smartScore.domain.StudentGrades;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Date;
import java.util.List;

public interface StudentGradesRepository extends JpaRepository<StudentGrades, Long> {

    @Query("SELECT s.score FROM StudentGrades s WHERE s.examDate = :examDate")
    List<Integer> findScoresByExamDate(@Param("examDate") Date examDate);


    @Query("""
            SELECT s.score
            FROM StudentGrades s
            WHERE s.examDate >= :start
            AND s.examDate < :end
            AND s.email = :email
            """)
    List<Integer> findScoresByDate(
            @Param("start") Date start,
            @Param("end") Date end,
            @Param("email") String email
    );

}