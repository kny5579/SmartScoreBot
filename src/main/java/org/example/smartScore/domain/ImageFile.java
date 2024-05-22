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
    private Date date;

    @Lob //db에 큰 값을 넣기 위함
    private byte[] data;

    @Builder
    public ImageFile(String imageName, Date date) {
        this.imageName = imageName;
        this.date = date;
    }
}