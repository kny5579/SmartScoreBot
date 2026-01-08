package org.example.smartScore.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class GradingRequest {
    private byte[] studentImage;
    private byte[] answerImage;
    private String studentId;
    private String fileName;
}

