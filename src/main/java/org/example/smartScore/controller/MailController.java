package org.example.smartScore.controller;

import lombok.RequiredArgsConstructor;
import org.example.smartScore.service.MailService;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
@RequiredArgsConstructor
public class MailController {

    private final MailService mailService;

    @ResponseBody
    @PostMapping("/mail")
    public ResponseEntity<String> sendMail(@RequestParam("mail") String mail) {
        int verificationNumber = mailService.sendMail(mail);
        return ResponseEntity.ok(String.valueOf(verificationNumber));
    }
}
