package org.example.smartScore.domain;

import jakarta.persistence.*;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.math.BigInteger;
import java.util.Date;

@Entity
@Table(name = "imagefile")
@Getter
@Setter
@NoArgsConstructor
public class ImageFile {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private BigInteger id;

    @Column(nullable = false)
    private String imageName;

    @Column(nullable = false)
    private Date examDate; // 시험 날짜

    @Column(nullable = false)
    private Date submitDate; // 제출 날짜

    @Lob //db에 큰 값을 넣기 위함
    private byte[] data;

    @Builder
    public ImageFile(String imageName, Date examDate, Date submitDate) {
        this.imageName = imageName;
        this.examDate = examDate;
        this.submitDate = submitDate;
    }
}