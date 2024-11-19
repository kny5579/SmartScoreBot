package org.example.smartScore.controller;

import org.example.smartScore.domain.ExcelFile;
import org.example.smartScore.repository.ExcelFileRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.List;

@Controller
public class UserViewController {

    @Autowired
    private ExcelFileRepository excelFileRepository;

    @GetMapping("/index")
    public String index(){
        return "index";
    }

    @GetMapping("/login")
    public String login(@RequestParam(value = "error", required = false)String error,
                        @RequestParam(value = "exception", required = false)String exception,
                        Model model){
        model.addAttribute("error", error);
        model.addAttribute("exception", exception);
        return "login";
    }

    @GetMapping("/resultList")
    public String getRecords(Model model) {
        List<ExcelFile> Records = excelFileRepository.findAll();
        model.addAttribute("Records", Records);
        return "resultList";
    }

    @GetMapping("/signup")
    public String signup(){
        return "signup";
    }
}
