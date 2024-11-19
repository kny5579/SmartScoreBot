package org.example.smartScore.domain;

import jakarta.persistence.*;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.math.BigInteger;
import java.sql.Timestamp;
import java.util.Date;

@Entity
@Getter
@Setter
@NoArgsConstructor
public class ExcelFile {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String fileName;

    @Column(name = "exam_date")
    private Date examDate; // 시험 날짜

    @Column(name = "submit_date")
    private Timestamp submitDate; // 제출 날짜

    @Lob //db에 큰 값을 넣기 위함
    private byte[] data;

    @Builder
    public ExcelFile(String fileName, Date examDate, Timestamp submitDate) {
        this.fileName = fileName;
        this.examDate = examDate;
        this.submitDate = submitDate;
    }
}