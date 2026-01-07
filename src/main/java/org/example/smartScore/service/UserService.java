package org.example.smartScore.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.domain.User;
import org.example.smartScore.dto.JoinDto;
import org.example.smartScore.repository.UserRepository;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@RequiredArgsConstructor
@Service
public class UserService {

    private final UserRepository userRepository;
    private final BCryptPasswordEncoder bCryptPasswordEncoder;

    @Transactional
    public Long save(JoinDto dto) {
        User user = User.builder()
                .email(dto.getEmail())
                .password(bCryptPasswordEncoder.encode(dto.getPassword()))
                .build();
        
        Long userId = userRepository.save(user).getId();
        log.info("User registered successfully: email={}, id={}", dto.getEmail(), userId);
        return userId;
    }

    @Transactional
    public boolean updatePassword(String email, String newPassword) {
        return userRepository.findByEmail(email)
                .map(user -> {
                    user.setPassword(bCryptPasswordEncoder.encode(newPassword));
                    userRepository.save(user);
                    log.info("Password updated successfully for user: {}", email);
                    return true;
                })
                .orElseGet(() -> {
                    log.warn("User not found for password update: {}", email);
                    return false;
                });
    }

    @Transactional
    public void deleteUser(String email) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new IllegalArgumentException("사용자를 찾을 수 없습니다: " + email));
        userRepository.delete(user);
        log.info("User deleted successfully: {}", email);
    }
}