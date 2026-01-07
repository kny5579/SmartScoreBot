package org.example.smartScore.constants;

public class AppConstants {
    
    private AppConstants() {
        throw new UnsupportedOperationException("Constants class");
    }

    public static final String FLASK_SERVER_URL = "http://flaskserver:5000/upload";
    public static final int BUFFER_SIZE = 1024;
    public static final String EXCEL_FILE_EXTENSION = ".xlsx";
    public static final String ZIP_FILE_NAME = "excel_files.zip";
}

