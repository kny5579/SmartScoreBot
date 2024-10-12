package org.example.smartScore.controller;

import lombok.RequiredArgsConstructor;
import org.example.smartScore.dto.JoinDto;
import org.example.smartScore.service.UserService;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;

@RequiredArgsConstructor
@Controller
public class UserController {

    private final UserService userService;

    @PostMapping("/user")
    public String signup(JoinDto request){
        userService.save(request); // 회원 가입 메소드 호출
        return "redirect:/login"; // 회원 가입이 완료된 후 로그인 페이지로 이동
    }
}
