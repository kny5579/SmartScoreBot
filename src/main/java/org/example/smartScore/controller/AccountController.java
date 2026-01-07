package org.example.smartScore.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.service.UserService;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Slf4j
@Controller
@RequestMapping("/delete-account")
@RequiredArgsConstructor
public class AccountController {

    private final UserService userService;

    @PostMapping
    public String deleteAccount(@AuthenticationPrincipal UserDetails userDetails) {
        String email = userDetails.getUsername();
        log.info("Account deletion requested for user: {}", email);

        userService.deleteUser(email);

        SecurityContextHolder.clearContext();
        log.info("Account deleted and session invalidated for user: {}", email);

        return "redirect:/";
    }
}
