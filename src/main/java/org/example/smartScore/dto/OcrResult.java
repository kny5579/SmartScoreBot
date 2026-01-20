package org.example.smartScore.dto;

import java.util.Map;

public record OcrResult(
        String studentId,
        Map<Integer, String> answers
) {}
