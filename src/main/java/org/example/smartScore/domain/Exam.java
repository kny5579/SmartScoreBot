package org.example.smartScore.domain;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.Date;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Exam { //시험 정보 저장
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String examNumber; //시험 번호

    @Column(nullable = false)
    private int examScore; //배점

    @Column(nullable = false)
    private Date examDate; //시험 날짜

}
