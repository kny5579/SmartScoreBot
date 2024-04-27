package org.example.smartScore.domain;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Student { //학생 정보 저장
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private Long studentNumber; //학번

    @Column(name = "score")
    private int totalScore; //점수

    @Builder
    Student(Long id, Long studentNumber, int totalScore) {
        this.id = id;
        this.studentNumber = studentNumber;
        this.totalScore = totalScore;
    }
}
