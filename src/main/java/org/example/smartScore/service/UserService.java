package org.example.smartScore.service;

import lombok.RequiredArgsConstructor;
import org.example.smartScore.domain.User;
import org.example.smartScore.dto.JoinDto;
import org.example.smartScore.repository.UserRepository;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.Optional;

@RequiredArgsConstructor
@Service
public class UserService {

    private final UserRepository userRepository;
    private final BCryptPasswordEncoder bCryptPasswordEncoder;

    public Long save(JoinDto dto){
        return userRepository.save(User.builder()
                .email(dto.getEmail())
                // 패스워드 암호화
                .password(bCryptPasswordEncoder.encode(dto.getPassword()))
                .build()).getId();
    }

    // 비밀번호 변경 메서드 추가
    public boolean updatePassword(String email, String newPassword) {
        Optional<User> userOptional = userRepository.findByEmail(email);
        if (userOptional.isPresent()) {
            User user = userOptional.get();
            user.setPassword(bCryptPasswordEncoder.encode(newPassword)); // 새 비밀번호 암호화 후 설정
            userRepository.save(user);
            return true;
        }
        return false; // 사용자를 찾을 수 없는 경우 false 반환
    }

}