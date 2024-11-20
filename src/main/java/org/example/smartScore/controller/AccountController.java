package org.example.smartScore.controller;

import org.example.smartScore.domain.User;
import org.example.smartScore.repository.UserRepository;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/delete-account")
public class AccountController {

    private final UserRepository userRepository;

    public AccountController(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @PostMapping
    public String deleteAccount(@AuthenticationPrincipal UserDetails userDetails) {
        // 현재 로그인한 사용자 정보 가져오기
        String email = userDetails.getUsername();

        // 데이터베이스에서 사용자 삭제
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new RuntimeException("사용자를 찾을 수 없습니다."));
        userRepository.delete(user);

        // Spring Security 세션 무효화
        SecurityContextHolder.clearContext();

        return "redirect:/"; // 메인 페이지로 리디렉션
    }
}
