package org.example.smartScore.config;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;
import org.example.smartScore.domain.User;
import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
public class LoginSuccessHandler implements AuthenticationSuccessHandler {

    @Override
    public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response,
                                        Authentication authentication) throws IOException {
        HttpSession session = request.getSession();

        // 인증된 사용자 정보 가져오기
        User user = (User) authentication.getPrincipal();

        // 세션에 이메일 저장
        session.setAttribute("email", user.getUsername());

        // 로그인 성공 후 리다이렉트
        response.sendRedirect("/");
    }
}