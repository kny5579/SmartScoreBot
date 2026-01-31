package org.example.smartScore.dto;

import java.util.Map;

public record OcrResult(
        String studentId,
        Map<String, String> answers,  // 문제 번호를 문자열로 변경 (소문제 지원: "15", "15-1", "15-2" 등)
        Map<String, Float> confidenceMap,  // 문제 번호별 confidence 값
        Map<String, Boolean> lowConfidenceMap  // 문제 번호별 low confidence 플래그
) {
    public OcrResult(String studentId, Map<String, String> answers) {
        this(studentId, answers, Map.of(), Map.of());
    }
    
    public OcrResult(String studentId, Map<String, String> answers, Map<String, Float> confidenceMap) {
        this(studentId, answers, confidenceMap, Map.of());
    }
}
