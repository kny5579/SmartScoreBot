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

    /**
     * 답안 타입
     */
    private enum AnswerType {
        NUMERIC,    // 숫자형 (정수)
        BOOLEAN,    // Boolean (true/false)
        STRING      // 문자열 (그 외)
    }
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
            
            boolean isCorrect = compareAnswersWithTypeCorrection(studentAnswer, correctAnswer);
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
            
            boolean isCorrect = compareAnswersWithTypeCorrection(studentAnswer, correctAnswer);
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

    /**
     * 답안 타입 기반 보정 비교
     * 
     * @param studentAnswer 학생 답안
     * @param correctAnswer 정답
     * @return 정답 여부
     */
    private boolean compareAnswersWithTypeCorrection(String studentAnswer, String correctAnswer) {
        if (studentAnswer == null || correctAnswer == null) {
            return false;
        }
        
        // 타입 판별
        AnswerType studentType = determineAnswerType(studentAnswer);
        AnswerType correctType = determineAnswerType(correctAnswer);
        
        // 타입이 다르면 무조건 오답
        if (studentType != correctType) {
            log.debug("Type mismatch: studentType={}, correctType={}, student={}, correct={}", 
                    studentType, correctType, studentAnswer, correctAnswer);
            return false;
        }
        
        // 타입별 보정 후 비교
        String correctedStudent = correctAnswerByType(studentAnswer, studentType);
        String correctedCorrect = correctAnswerByType(correctAnswer, correctType);
        
        boolean isEqual = correctedStudent.equals(correctedCorrect);
        
        if (!isEqual) {
            log.debug("Answer mismatch after correction: student={} -> {}, correct={} -> {}", 
                    studentAnswer, correctedStudent, correctAnswer, correctedCorrect);
        }
        
        return isEqual;
    }

    /**
     * 답안 타입 판별
     * 
     * @param answer 답안
     * @return 답안 타입
     */
    private AnswerType determineAnswerType(String answer) {
        if (answer == null || answer.trim().isEmpty()) {
            return AnswerType.STRING;
        }
        
        String trimmed = answer.trim();
        
        // Boolean 타입 확인 (OCR 보정 후 true/false로 정규화 가능한지)
        String booleanCorrected = correctBooleanAnswer(trimmed);
        if (booleanCorrected != null) {
            return AnswerType.BOOLEAN;
        }
        
        // 숫자형 확인 (OCR 보정 후 숫자만 남았을 때 값이 존재하면 숫자형)
        String numericCorrected = correctNumericAnswer(trimmed);
        if (numericCorrected != null && !numericCorrected.isEmpty()) {
            return AnswerType.NUMERIC;
        }
        
        // 나머지는 문자열
        return AnswerType.STRING;
    }

    /**
     * 타입별 답안 보정
     * 
     * @param answer 원본 답안
     * @param type 답안 타입
     * @return 보정된 답안
     */
    private String correctAnswerByType(String answer, AnswerType type) {
        if (answer == null) {
            return "";
        }
        
        switch (type) {
            case NUMERIC:
                String numericResult = correctNumericAnswer(answer);
                return numericResult != null ? numericResult : "";
            case BOOLEAN:
                String booleanResult = correctBooleanAnswer(answer);
                return booleanResult != null ? booleanResult : "";
            case STRING:
                return correctStringAnswer(answer);
            default:
                return answer.trim();
        }
    }

    /**
     * 숫자형 답안 보정
     * 
     * 보정 규칙:
     * - "나" -> "4"
     * - "|", "Ⅰ", "l", "!" -> "1"
     * - "O", "o" -> "0"
     * - "S" -> "5"
     * - "B" -> "8"
     * - 숫자가 아닌 문자는 제거
     * - 결과가 비어있으면 null 반환
     */
    private String correctNumericAnswer(String answer) {
        if (answer == null || answer.trim().isEmpty()) {
            return null;
        }
        
        String corrected = answer.trim();
        
        // 특수 문자 보정
        corrected = corrected.replace("나", "4");
        corrected = corrected.replace("|", "1");
        corrected = corrected.replace("Ⅰ", "1");
        corrected = corrected.replace("l", "1");
        corrected = corrected.replace("!", "1");
        corrected = corrected.replace("O", "0");
        corrected = corrected.replace("o", "0");
        corrected = corrected.replace("S", "5");
        corrected = corrected.replace("B", "8");
        
        // 숫자가 아닌 문자 제거
        corrected = corrected.replaceAll("[^0-9]", "");
        
        // 결과가 비어있으면 숫자형 아님
        if (corrected.isEmpty()) {
            return null;
        }
        
        return corrected;
    }

    /**
     * Boolean 답안 보정
     * 
     * 보정 규칙:
     * - 대소문자 무시
     * - true/false로만 정규화
     * - 1, O -> true
     * - 0, X -> false
     * - 정규화 실패 시 null 반환
     */
    private String correctBooleanAnswer(String answer) {
        if (answer == null || answer.trim().isEmpty()) {
            return null;
        }
        
        String trimmed = answer.trim();
        String lower = trimmed.toLowerCase();
        
        // true 패턴
        if (lower.equals("true") || lower.equals("t") || 
            trimmed.equals("1") || trimmed.equals("O") || trimmed.equals("o")) {
            return "true";
        }
        
        // false 패턴
        if (lower.equals("false") || lower.equals("f") || 
            trimmed.equals("0") || trimmed.equals("X") || trimmed.equals("x")) {
            return "false";
        }
        
        // 정규화 실패
        return null;
    }

    /**
     * 문자열 답안 보정
     * 
     * 보정 규칙:
     * - 앞의 "답:" 제거
     * - 공백 제거
     * - 나머지는 그대로 비교
     */
    private String correctStringAnswer(String answer) {
        if (answer == null) {
            return "";
        }
        
        String corrected = answer.trim();
        
        // 앞의 "답:" 제거
        if (corrected.startsWith("답:")) {
            corrected = corrected.substring(2).trim();
        }
        
        // 공백 제거
        corrected = corrected.replaceAll("\\s+", "");
        
        return corrected;
    }

    public record GradingResult(
            String studentId,
            int score,
            int correctCount,
            int totalQuestions,
            Map<String, Boolean> questionResults  // 문제 번호를 문자열로 변경 (소문제 지원)
    ) {}
}

