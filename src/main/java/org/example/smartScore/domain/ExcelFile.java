package org.example.smartScore.domain;

import jakarta.persistence.*;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.math.BigInteger;
import java.util.Date;

@Entity
@Getter
@Setter
@NoArgsConstructor
public class ExcelFile {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private BigInteger id;

    @Column(nullable = false)
    private String fileName;

    @Column(nullable = false)
    private Date date;

    @Lob //db에 큰 값을 넣기 위함
    private byte[] data;

    @Builder
    public ExcelFile(String fileName, Date date) {
        this.fileName = fileName;
        this.date = date;
    }
}