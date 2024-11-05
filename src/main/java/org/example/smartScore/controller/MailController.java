package org.example.smartScore.controller;

import org.example.smartScore.service.MailService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class MailController {
    private final MailService mailService;
    private int number; // 이메일 인증 숫자를 저장하는 변수

    public MailController(MailService mailService) {
        this.mailService = mailService;
    }

    // 인증 이메일 전송
    @PostMapping("/mailSend")
    public String mailSend(@RequestParam String mail, Model model) {
        try {
            number = mailService.sendMail(mail);
            model.addAttribute("message", "인증 번호가 이메일로 전송되었습니다.");
        } catch (Exception e) {
            model.addAttribute("error", "인증 번호 전송에 실패했습니다. 다시 시도해주세요.");
        }

        return "signup"; // 뷰 이름을 "signup"으로 변경
    }

    // 인증번호 일치여부 확인
    @GetMapping("/mailCheck")
    public String mailCheck(@RequestParam String userNumber, Model model) {
        boolean isMatch = userNumber.equals(String.valueOf(number));
        if (isMatch) {
            model.addAttribute("message", "인증이 완료되었습니다.");
        } else {
            model.addAttribute("error", "인증 번호가 일치하지 않습니다. 다시 확인해주세요.");
        }

        return "signup"; // 뷰 이름을 "signup"으로 변경
    }
}
