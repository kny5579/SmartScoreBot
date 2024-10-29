FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# FLASK_APP 환경 변수를 설정
ENV FLASK_APP=app.py

# Flask 애플리케이션을 실행
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
