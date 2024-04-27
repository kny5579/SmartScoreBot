package org.example.smartScore.domain;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class StudentAnswer { //학생 답안 저장
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "student_id", nullable = false)
    private Student student; //학생 정보

    @ManyToOne
    @JoinColumn(name = "exam_id", nullable = false)
    private Exam exam; //시험 정보

    @Column(nullable = false)
    private String studentAnswer; //학생 답안

}
