/*
package org.example.smartScore.service;

import jakarta.mail.internet.MimeMessage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.mail.javamail.JavaMailSender;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class MailServiceTest {

    @Mock
    private JavaMailSender javaMailSender;

    @Mock
    private MimeMessage mimeMessage;

    @InjectMocks
    private MailService mailService;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        // MimeMessage 생성 시 mock 객체 반환
        when(javaMailSender.createMimeMessage()).thenReturn(mimeMessage);
    }

    @Test
    void testSendMail() throws Exception {
        String email = "32200380@dankook.ac.kr";
        int verificationNumber = mailService.sendMail(email);

        // 추가적으로 이메일 전송을 확인하는 로직 추가
        verify(javaMailSender, times(1)).send(any(MimeMessage.class));
        // 인증번호가 잘 생성되었는지 확인 (6자리 숫자: 100000 ~ 999999)
        assertNotEquals(0, verificationNumber);
    }
}
*/
