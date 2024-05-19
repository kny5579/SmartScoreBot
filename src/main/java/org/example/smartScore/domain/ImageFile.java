package org.example.smartScore.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.math.BigInteger;

@Entity
@Table(name = "imagefile")
@Getter
@NoArgsConstructor
public class ImageFile {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private BigInteger id;

    @Column(nullable = false)
    private String imageName;

    @Column(nullable = false)
    private String date;

    public ImageFile(String imageName, String date) {
        this.imageName = imageName;
        this.date = date;
    }
}
