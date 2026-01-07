package org.example.smartScore.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.service.MailService;
import org.example.smartScore.service.UserService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@Controller
@RequiredArgsConstructor
public class ForgotPasswordController {

    private final MailService mailService;
    private final UserService userService;
    private final Map<String, Integer> passwordResetVerificationCodes = new HashMap<>();

    @GetMapping("/forgot-password")
    public String showForgotPasswordPage() {
        return "forgot-password";
    }

    @PostMapping("/password-reset-mail")
    public ResponseEntity<String> sendPasswordResetVerificationCode(@RequestParam("email") String email) {
        int generatedNumber = mailService.sendMail(email);
        passwordResetVerificationCodes.put(email, generatedNumber);
        log.info("비밀번호 찾기 인증번호 생성 및 저장: 이메일={}", email);
        return ResponseEntity.ok("인증번호가 전송되었습니다.");
    }

    @PostMapping("/verify-password-reset-code")
    public ResponseEntity<String> verifyPasswordResetCode(@RequestBody Map<String, Object> requestData) {
        String email = (String) requestData.get("email");
        int enteredCode = (int) requestData.get("enteredCode");

        Integer storedCode = passwordResetVerificationCodes.get(email);
        log.debug("인증번호 검증: 이메일={}, 입력된 인증번호={}", email, enteredCode);

        if (storedCode != null && storedCode.equals(enteredCode)) {
            passwordResetVerificationCodes.remove(email);
            return ResponseEntity.ok("success");
        } else {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body("인증 번호가 일치하지 않습니다.");
        }
    }

    @PostMapping("/reset-password")
    public ResponseEntity<String> resetPassword(@RequestParam("email") String email,
                                                @RequestParam("newPassword") String newPassword,
                                                @RequestParam("confirmPassword") String confirmPassword) {
        if (!newPassword.equals(confirmPassword)) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body("비밀번호가 일치하지 않습니다.");
        }

        boolean isUpdated = userService.updatePassword(email, newPassword);
        if (isUpdated) {
            log.info("비밀번호 변경 성공: 이메일={}", email);
            return ResponseEntity.ok("비밀번호가 성공적으로 변경되었습니다.");
        } else {
            log.warn("비밀번호 변경 실패: 이메일={}", email);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("비밀번호 변경에 실패했습니다.");
        }
    }
}
