FROM openjdk:19

WORKDIR /app

COPY build/libs/smart_score-0.0.1-SNAPSHOT.jar app.jar

ENTRYPOINT ["java", "-jar", "app.jar"]