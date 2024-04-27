package org.example.smartScore.domain;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class CorrectAnswer { //정답지 정보 저장
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "exam_id", nullable = false)
    private Exam exam; //시험 정보

    @Column(name = "correctAnswer", nullable = false)
    private Long correctAnswer; //정답

    @Builder
    CorrectAnswer(Long id, Exam exam, Long correctAnswer) {
        this.id = id;
        this.exam = exam;
        this.correctAnswer = correctAnswer;
    }

}
