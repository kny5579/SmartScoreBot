-- 데이터베이스가 이미 존재하지 않으면 생성
CREATE DATABASE IF NOT EXISTS smartscoredb;

-- 해당 데이터베이스 사용
USE smartscoredb;

-- 기존 테이블이 있다면 삭제
DROP TABLE IF EXISTS imagefile;
DROP TABLE IF EXISTS excel_file;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS student_grades;

-- 새로운 테이블 생성

CREATE TABLE IF NOT EXISTS imagefile (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    exam_date DATE,
    submit_date DATE,
    data LONGBLOB
);

CREATE TABLE IF NOT EXISTS excel_file (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    exam_date DATE,
    submit_date DATE,
    data LONGBLOB
);

CREATE TABLE IF NOT EXISTS users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS student_grades (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    score INT NOT NULL,
    exam_date DATE,
    student_id VARCHAR(255) NOT NULL
);
