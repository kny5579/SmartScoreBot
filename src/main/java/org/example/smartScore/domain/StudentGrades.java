package org.example.smartScore.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.math.BigInteger;
import java.util.Date;

@Entity
@Table(name = "student_grades")
@Getter
@Setter
@NoArgsConstructor
public class StudentGrades {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private BigInteger id;

    @Column(nullable = false)
    private String studentId; // 학번

    @Column(nullable = false)
    private int score; // 점수

    @Column(name = "exam_date")
    private Date examDate;



}