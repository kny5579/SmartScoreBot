package org.example.smartScore.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.util.Date;

@Entity
@Getter
@Setter
public class ProcessedFileEntity { //플라스크에서 넘어온 파일 정보
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String filename;

    @Lob //db에 큰 값을 넣기 위함
    private byte[] data;

    private String fileType;

    @Temporal(TemporalType.DATE)
    private Date date;
}