package org.example.smartScore.controller;

import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.repository.ImageFileRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;
import org.springframework.util.StringUtils;
import org.springframework.core.io.ByteArrayResource;

import java.io.IOException;
import java.util.Optional;

@Controller
@RequestMapping("/images")
public class ImageController {

    @Autowired
    private ImageFileRepository imageFileRepository;

    @GetMapping("/{id}")
    @ResponseBody
    public ResponseEntity<Resource> getImage(@PathVariable Long id) throws IOException {
        Optional<ImageFile> imageFile = imageFileRepository.findById(id);

        byte[] imageData = imageFile.get().getData();

        Resource resource = new ByteArrayResource(imageData);

        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_JPEG)
                .body(resource);
    }
}
