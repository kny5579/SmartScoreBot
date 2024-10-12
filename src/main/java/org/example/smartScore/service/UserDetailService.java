package org.example.smartScore.service;

import lombok.RequiredArgsConstructor;
import org.example.smartScore.domain.User;
import org.example.smartScore.repository.UserRepository;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.stereotype.Service;

@RequiredArgsConstructor
@Service
public class UserDetailService implements UserDetailsService {
    private final UserRepository userRepository;

    @Override //사용자 이메일을 통해 사용자 정보 불러오는 메소드
    public User loadUserByUsername(String email){
        return userRepository.findByEmail(email)
                .orElseThrow(() -> new IllegalArgumentException((email)));
    }
}
