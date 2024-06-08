## 스마트 채점봇

---

docker run -p 8080:8080 docker-springboot

docker run -d --name flaskserver -p 5000:5000 --network smartscorebot_default flask-server


---
# 도커 mysql 테이블 설정

docker exec -it mysqldb mysql -u root -p

mysql> use smartscoredb

mysql> CREATE TABLE excel_file (
id BIGINT AUTO_INCREMENT PRIMARY KEY,
file_name VARCHAR(255) NOT NULL,
date DATE NOT NULL,
data LONGBLOB
);

mysql> CREATE TABLE imagefile ( 
id BIGINT AUTO_INCREMENT PRIMARY KEY,
image_name VARCHAR(255) NOT NULL,
date DATE NOT NULL, 
data LONGBLOB
);

mysql> exit
