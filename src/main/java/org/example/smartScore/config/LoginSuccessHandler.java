package org.example.smartScore.config;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.domain.User;
import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Slf4j
@Component
public class LoginSuccessHandler implements AuthenticationSuccessHandler {

    private static final String SESSION_EMAIL_ATTRIBUTE = "email";
    private static final String REDIRECT_URL = "/";

    @Override
    public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response,
                                        Authentication authentication) throws IOException {
        HttpSession session = request.getSession();
        User user = (User) authentication.getPrincipal();
        
        session.setAttribute(SESSION_EMAIL_ATTRIBUTE, user.getUsername());
        log.info("User logged in successfully: {}", user.getUsername());

        response.sendRedirect(REDIRECT_URL);
    }
}