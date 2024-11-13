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
    private final Map<String, Integer> passwordResetVerificationCodes = new HashMap<>();

    @GetMapping("/forgot-password")
    public String showForgotPasswordPage() {
        return "forgot-password";
    }

    // 비밀번호 찾기 인증 메일 전송 - 변경된 경로 /password-reset-mail
    @PostMapping("/password-reset-mail")
    public ResponseEntity<String> sendPasswordResetVerificationCode(@RequestParam("email") String email) {
        int generatedNumber = mailService.sendMail(email);
        passwordResetVerificationCodes.put(email, generatedNumber);
        logger.info("비밀번호 찾기 인증번호 생성 및 저장: 이메일={}, 인증번호={}", email, generatedNumber);
        return ResponseEntity.ok("인증번호가 전송되었습니다.");
    }

    // 비밀번호 찾기 인증번호 검증 - 변경된 경로 /verify-password-reset-code
    @PostMapping("/verify-password-reset-code")
    public ResponseEntity<String> verifyPasswordResetCode(@RequestBody Map<String, Object> requestData) {
        String email = (String) requestData.get("email");
        int enteredCode = (int) requestData.get("enteredCode");

        Integer storedCode = passwordResetVerificationCodes.get(email);
        logger.info("인증번호 검증: 이메일={}, 입력된 인증번호={}, 저장된 인증번호={}", email, enteredCode, storedCode);

        if (storedCode != null && storedCode.equals(enteredCode)) {
            passwordResetVerificationCodes.remove(email);
            return ResponseEntity.ok("success");
        } else {
            return ResponseEntity.status(400).body("인증 번호가 일치하지 않습니다.");
        }
    }

    // 비밀번호 변경 처리
    @PostMapping("/reset-password")
    public ResponseEntity<String> resetPassword(@RequestParam("email") String email,
                                                @RequestParam("newPassword") String newPassword,
                                                @RequestParam("confirmPassword") String confirmPassword) {
        if (!newPassword.equals(confirmPassword)) {
            return ResponseEntity.status(400).body("비밀번호가 일치하지 않습니다.");
        }

        boolean isUpdated = userService.updatePassword(email, newPassword);
        if (isUpdated) {
            return ResponseEntity.ok("비밀번호가 성공적으로 변경되었습니다.");
        } else {
            return ResponseEntity.status(500).body("비밀번호 변경에 실패했습니다.");
        }
    }
}
