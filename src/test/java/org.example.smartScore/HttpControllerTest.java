package org.example.smartScore;

import org.example.smartScore.controller.HttpController;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.web.client.RestTemplate;

import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(HttpController.class)
class HttpControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private RestTemplate restTemplate;

    @Test
    void uploadImages() throws Exception {
        // 테스트에 필요한 임의의 이미지 파일 생성
        MockMultipartFile file1 = new MockMultipartFile("file", "test-image1.jpg", MediaType.IMAGE_JPEG_VALUE, "image1".getBytes());
        MockMultipartFile file2 = new MockMultipartFile("file", "test-image2.jpg", MediaType.IMAGE_JPEG_VALUE, "image2".getBytes());

        // 이미지 업로드와 날짜 데이터를 포함하여 POST
        ResultActions resultActions = mockMvc.perform(MockMvcRequestBuilders.multipart("/upload")
                .file(file1)
                .file(file2)
                .param("date", "2022-05-12"));

        // 응답 확인
        resultActions.andExpect(status().isOk())
                .andExpect(view().name("result"))
                .andExpect(model().attributeExists("excelFileBytes"));
    }

    @Test
    void downloadExcelFile() throws Exception {
        // 엑셀 파일 다운로드 요청 보내기
        ResultActions resultActions = mockMvc.perform(MockMvcRequestBuilders.get("/download")
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .content(new byte[0]));

        // 응답 확인
        resultActions.andExpect(status().isOk())
                .andExpect(header().exists("Content-Disposition"))
                .andExpect(header().string("Content-Disposition", "attachment; filename=excel_file.xlsx"))
                .andExpect(content().contentType(MediaType.APPLICATION_OCTET_STREAM));
    }
}
