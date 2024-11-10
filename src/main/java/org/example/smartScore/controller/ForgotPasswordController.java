package org.example.smartScore.controller;

import lombok.RequiredArgsConstructor;
import org.example.smartScore.service.MailService;
import org.example.smartScore.service.UserService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@Controller
@RequiredArgsConstructor
public class ForgotPasswordController {

    private static final Logger logger = LoggerFactory.getLogger(ForgotPasswordController.class);
    private final MailService mailService;
    private final UserService userService;
    private final Map<String, Integer> verificationCodes = new HashMap<>(); // 이메일별 인증 코드 저장

    // 비밀번호 찾기 페이지 매핑
    @GetMapping("/forgot-password")
    public String showForgotPasswordPage() {
        return "forgot-password"; // `forgot-password.html` 파일 이름과 일치해야 함
    }

    // 이메일 인증 번호 전송 및 저장
    @PostMapping("/send-verification-code")
    public ResponseEntity<String> sendVerificationCode(@RequestParam("email") String email) {
        int generatedNumber = mailService.sendMail(email);
        verificationCodes.put(email, generatedNumber); // 이메일과 함께 인증 코드 저장
        logger.info("인증번호 생성: 이메일={}, 인증번호={}", email, generatedNumber);
        return ResponseEntity.ok("인증번호가 전송되었습니다.");
    }

    // 이메일 인증 번호 검증 엔드포인트
    @PostMapping("/verify-code")
    public ResponseEntity<String> verifyCode(@RequestParam("email") String email,
                                             @RequestParam("enteredCode") int enteredCode) {
        Integer storedCode = verificationCodes.get(email);
        if (storedCode != null) {
            logger.info("입력된 인증번호={}, 서버 저장 인증번호={}", enteredCode, storedCode);
            if (storedCode == enteredCode) {
                logger.info("인증번호 일치: 이메일={}, 인증번호={}", email, enteredCode);
                return ResponseEntity.ok("success");
            } else {
                logger.warn("인증 번호가 일치하지 않습니다. 입력된 번호={}, 저장된 번호={}", enteredCode, storedCode);
                return ResponseEntity.status(400).body("인증 번호가 일치하지 않습니다.");
            }
        } else {
            logger.warn("이메일에 대한 인증 번호를 찾을 수 없습니다: 이메일={}", email);
            return ResponseEntity.status(400).body("인증 번호를 찾을 수 없습니다.");
        }
    }

    // 비밀번호 변경 처리
    @PostMapping("/reset-password")
    public ResponseEntity<String> resetPassword(@RequestParam("email") String email,
                                                @RequestParam("newPassword") String newPassword,
                                                @RequestParam("confirmPassword") String confirmPassword) {
        if (!newPassword.equals(confirmPassword)) {
            logger.warn("비밀번호가 일치하지 않습니다: 이메일={}", email);
            return ResponseEntity.status(400).body("비밀번호가 일치하지 않습니다. 다시 확인해주세요.");
        }

        // 비밀번호 업데이트 처리
        boolean isUpdated = userService.updatePassword(email, newPassword);
        if (isUpdated) {
            verificationCodes.remove(email); // 비밀번호 변경 후 인증번호 제거
            logger.info("비밀번호가 성공적으로 변경되었습니다: 이메일={}", email);
            return ResponseEntity.ok("비밀번호가 성공적으로 변경되었습니다.");
        } else {
            logger.error("비밀번호 변경 실패: 이메일={}", email);
            return ResponseEntity.status(500).body("비밀번호 변경에 실패했습니다. 다시 시도해주세요.");
        }
    }
}
