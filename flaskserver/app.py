from flask import Flask, render_template, request, send_file, url_for
import zipfile
import os
from io import BytesIO
from PIL import Image
import pandas as pd
from datetime import datetime, timezone

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('Results.html')
#
# @app.route('/')
# def home():
#     return '작심삼일 안녕!'


@app.route('/upload', methods=['POST'])
def upload_files():
    # 이미지 받기
    student_files = request.files.getlist("student_files")
    answer_files = request.files.getlist("answer_files")

    # 날짜 받기
    date = request.form.get("date")

    # 날짜 포맷팅
    formatted_date = datetime.fromtimestamp(int(date) / 1000, timezone.utc).strftime("%Y-%m-%d")

    # 날짜별 저장 경로
    upload_folder = f'./sample_data/{formatted_date}/'

    # 디렉토리 없으면 생성
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # 기존의 사진 삭제
    for f in os.listdir(upload_folder):
        try:
            os.remove(os.path.join(upload_folder, f))
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    # 받은 사진 저장
    for f in student_files + answer_files:
        f.save(os.path.join(upload_folder, f.filename))

    # 파일 처리 함수 호출 (여기에 파일 처리 로직을 작성)
    process_result(upload_folder)

    # ZIP 파일 생성
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for root, _, files in os.walk(upload_folder):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, upload_folder))

    zip_buffer.seek(0)

    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f'processed_files_{formatted_date}.zip')


def process_result(upload_folder):
    # 샘플 엑셀 파일 생성
    excel_file_path = os.path.join(upload_folder, 'sample.xlsx')
    df = pd.DataFrame({'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']})
    df.to_excel(excel_file_path, index=False)

    # 샘플 이미지 파일 생성
    image_file_path = os.path.join(upload_folder, 'sample.jpg')
    img = Image.new('RGB', (100, 100), color=(73, 109, 137))
    img.save(image_file_path)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
