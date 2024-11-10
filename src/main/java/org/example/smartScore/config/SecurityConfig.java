package org.example.smartScore.config;

import lombok.RequiredArgsConstructor;
import org.example.smartScore.service.UserDetailService;
import org.springframework.boot.autoconfigure.security.servlet.PathRequest;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityCustomizer;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.AuthenticationFailureHandler;

@EnableWebSecurity
@EnableMethodSecurity
@Configuration
@RequiredArgsConstructor
public class SecurityConfig {

    private final UserDetailService userService;
    private final AuthenticationFailureHandler customFailureHandler;

    // 특정 HTTP 요청에 대한 웹 기반 보안 구성
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        return http
                .authorizeRequests() // 인증, 인가 설정
                .requestMatchers("/images/**").permitAll()
                .requestMatchers("/", "/index", "/login", "/signup", "/user", "/mail/**", "/forgot-password").permitAll()
                .anyRequest().authenticated()
                .and()
                .formLogin() // 폼 기반 로그인 설정
                .loginPage("/login")
                .defaultSuccessUrl("/", true)
                .failureHandler(customFailureHandler) // 로그인 실패 핸들러
                .and()
                .logout() // 로그아웃 설정
                .logoutSuccessUrl("/")
                .invalidateHttpSession(true)
                .and()
                .rememberMe() //자동 로그인 설정
                .key("SecretKey") // Remember Me에 사용할 고유 키
                .rememberMeParameter("remember-me")
                .tokenValiditySeconds(60*60*24*3) // 쿠키 유효 기간 설정 (3일)
                .and()
                .csrf().disable() // CSRF 비활성화 (실습용)
                .build();
    }

    // 인증 관리자 관련 설정
    @Bean
    public AuthenticationManager authenticationManager(HttpSecurity http, BCryptPasswordEncoder bCryptPasswordEncoder, UserDetailService userDetailService)
            throws Exception{
        return http.getSharedObject(AuthenticationManagerBuilder.class)
                .userDetailsService(userService) // 사용자 정보 서비스 설정
                .passwordEncoder(bCryptPasswordEncoder)
                .and()
                .build();
    }

    @Bean
    public WebSecurityCustomizer webSecurityCustomizer() {
        return (web) -> web.ignoring().requestMatchers(
                "/static/**", "/css/**", "/js/**", "/images/**", "/index.html");
    }

    // 패스워드 인코더로 사용할 빈 등록
    @Bean
    public BCryptPasswordEncoder bCryptPasswordEncoder(){
        return new BCryptPasswordEncoder();
    }
}
