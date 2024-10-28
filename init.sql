    CREATE DATABASE IF NOT EXISTS smartscoredb;

    USE smartscoredb;

    CREATE TABLE IF NOT EXISTS imagefile (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        image_name VARCHAR(255) NOT NULL,
        date DATE NOT NULL,
        data LONGBLOB
        );

    CREATE TABLE IF NOT EXISTS excel_file (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        file_name VARCHAR(255) NOT NULL,
        date DATE NOT NULL,
        data LONGBLOB
        );

    CREATE TABLE IF NOT EXISTS users (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL
        );

    CREATE TABLE IF NOT EXISTS StudentGrades (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        score INT NOT NULL,
        date DATE NOT NULL
    )