package org.example.smartScore.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.math.BigInteger;

@Entity
@Getter
@NoArgsConstructor
public class ExcelFile {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private BigInteger id;

    @Column(nullable = false)
    private String fileName;

    @Column(nullable = false)
    private String date;

    public ExcelFile(String fileName, String date) {
        this.fileName = fileName;
        this.date = date;
    }
}