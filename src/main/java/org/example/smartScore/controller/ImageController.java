package org.example.smartScore.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.smartScore.domain.ImageFile;
import org.example.smartScore.repository.ImageFileRepository;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Slf4j
@Controller
@RequestMapping("/images")
@RequiredArgsConstructor
public class ImageController {

    private final ImageFileRepository imageFileRepository;

    @GetMapping("/{id}")
    @ResponseBody
    public ResponseEntity<Resource> getImage(@PathVariable Long id) {
        ImageFile imageFile = imageFileRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Image not found with id: " + id));

        Resource resource = new ByteArrayResource(imageFile.getData());

        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_JPEG)
                .body(resource);
    }
}
