package org.example.smartScore.service;

import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class MailService {

    private static final Logger logger = LoggerFactory.getLogger(MailService.class);
    private final JavaMailSender javaMailSender;
    private static final String senderEmail = "wisejohn950330@gmail.com";
    private int number; // 인증번호를 저장할 인스턴스 변수

    // 인증번호 생성 메서드
    public void createNumber() {
        number = (int)(Math.random() * 900000) + 100000; // 6자리 인증번호 생성
        logger.info("Generated verification number: {}", number); // 인증번호를 로그에 출력
    }

    // 이메일 메시지 생성 메서드
    public MimeMessage createMail(String mail) {
        createNumber(); // 인증번호 생성
        MimeMessage message = javaMailSender.createMimeMessage();

        try {
            message.setFrom(senderEmail);
            message.setRecipients(MimeMessage.RecipientType.TO, mail);
            message.setSubject("이메일 인증");
            String body = "<h3>요청하신 인증 번호입니다.</h3>" +
                    "<h1>" + number + "</h1>" +
                    "<h3>감사합니다.</h3>";
            message.setText(body, "UTF-8", "html");
        } catch (MessagingException e) {
            logger.error("Failed to create email message", e);
        }

        return message;
    }

    // 인증 메일 발송 메서드
    public int sendMail(String mail) {
        MimeMessage message = createMail(mail);
        javaMailSender.send(message);
        return number; // 생성된 인증번호 반환
    }
}
