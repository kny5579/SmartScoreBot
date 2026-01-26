package org.example.smartScore.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.domain.StudentGrades;
import org.example.smartScore.repository.StudentGradesRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;

@Slf4j
@Service
@RequiredArgsConstructor
public class GradingService {

    private final StudentGradesRepository studentGradesRepository;
    public GradingResult gradeAnswersWithQuestionNumbers(
            Map<String, String> studentAnswers, 
            Map<String, String> correctAnswers,
            String studentId, Date examDate, String email) {
        
        // 모든 문제 번호 수집 (학생 답안과 정답 모두 포함)
        Set<String> allQuestionNumbers = new LinkedHashSet<>();
        allQuestionNumbers.addAll(correctAnswers.keySet());
        allQuestionNumbers.addAll(studentAnswers.keySet());
        
        int totalQuestions = allQuestionNumbers.size();
        int correctCount = 0;
        Map<String, Boolean> questionResults = new LinkedHashMap<>();
        
        // 문제 번호별로 채점 (소문제 포함)
        for (String questionNumber : allQuestionNumbers) {
            String studentAnswer = studentAnswers.getOrDefault(questionNumber, "").trim();
            String correctAnswer = correctAnswers.getOrDefault(questionNumber, "").trim();
            
            boolean isCorrect = studentAnswer.equals(correctAnswer);
            if (isCorrect && !studentAnswer.isEmpty()) {
                correctCount++;
            }
            questionResults.put(questionNumber, isCorrect);
            
            log.debug("Question {}: student={}, correct={}, result={}", 
                    questionNumber, studentAnswer, correctAnswer, isCorrect ? "O" : "X");
        }
        
        int score = totalQuestions > 0 ? 
                (int) Math.round((double) correctCount / totalQuestions * 100) : 0;
        
        // 점수 저장
        saveGrade(studentId, score, examDate, email);
        
        log.info("Grading completed for student {}: {}/{} correct, score: {}", 
                studentId, correctCount, totalQuestions, score);
        
        return new GradingResult(studentId, score, correctCount, totalQuestions, questionResults);
    }

    public GradingResult gradeAnswers(List<String> studentAnswers, List<String> correctAnswers, 
                                     String studentId, Date examDate, String email) {
        
        if (studentAnswers.size() != correctAnswers.size()) {
            log.warn("Answer count mismatch: student={}, correct={}", 
                    studentAnswers.size(), correctAnswers.size());
        }
        
        int totalQuestions = Math.max(studentAnswers.size(), correctAnswers.size());
        int correctCount = 0;
        Map<String, Boolean> questionResults = new LinkedHashMap<>();
        
        for (int i = 0; i < totalQuestions; i++) {
            String studentAnswer = i < studentAnswers.size() ? 
                    studentAnswers.get(i).trim() : "";
            String correctAnswer = i < correctAnswers.size() ? 
                    correctAnswers.get(i).trim() : "";
            
            boolean isCorrect = studentAnswer.equals(correctAnswer);
            if (isCorrect) {
                correctCount++;
            }
            questionResults.put(String.valueOf(i + 1), isCorrect);
        }
        
        int score = totalQuestions > 0 ? 
                (int) Math.round((double) correctCount / totalQuestions * 100) : 0;

        saveGrade(studentId, score, examDate, email);
        
        log.info("Grading completed for student {}: {}/{} correct, score: {}", 
                studentId, correctCount, totalQuestions, score);
        
        return new GradingResult(studentId, score, correctCount, totalQuestions, questionResults);
    }

    @Transactional
    private void saveGrade(String studentId, int score, Date examDate, String email) {
        StudentGrades grade = new StudentGrades();
        grade.setStudentId(studentId);
        grade.setScore(score);
        grade.setExamDate(examDate);
        grade.setEmail(email);
        studentGradesRepository.save(grade);
    }

    public record GradingResult(
            String studentId,
            int score,
            int correctCount,
            int totalQuestions,
            Map<String, Boolean> questionResults  // 문제 번호를 문자열로 변경 (소문제 지원)
    ) {}
}

