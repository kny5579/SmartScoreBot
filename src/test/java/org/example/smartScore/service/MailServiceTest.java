package org.example.smartScore.service;

import jakarta.mail.internet.MimeMessage;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.mail.javamail.JavaMailSender;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.mockito.Mockito.*;

class MailServiceTest {

    @Mock
    private JavaMailSender javaMailSender;

    @InjectMocks
    private MailService mailService;

    public MailServiceTest() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testSendMail() throws Exception {
        String email = "32200380@dankook.ac.kr";
        int verificationNumber = mailService.sendMail(email);

        // 추가적으로 이메일 전송을 확인하는 로직 추가
        verify(javaMailSender, times(1)).send((MimeMessage) any());
        // 인증번호가 잘 생성되었는지 확인
        assertNotEquals(0, verificationNumber);
    }
}
