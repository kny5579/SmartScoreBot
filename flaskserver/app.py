import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pytesseract import Output

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from keras.utils import load_img, img_to_array

from flask import Flask, render_template, request



app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('Results.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    # 이미지 받기
    student_files = request.files.getlist("student_files")
    answer_files = request.files.getlist("answer_files")

    # 날짜 받기
    exam_date = request.form.get("exam_date")

    # 날짜 포맷팅
    formatted_exam_date = datetime.fromtimestamp(int(exam_date) / 1000, timezone.utc).strftime("%Y-%m-%d")

    # 날짜별 저장 경로
    upload_folder = f'./sample_data/{formatted_exam_date}/'

    # 기존의 사진 삭제
    if os.path.exists(upload_folder):
        for f in os.listdir(upload_folder):
            try:
                os.remove(os.path.join(upload_folder, f))
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
    else:
        os.makedirs(upload_folder, exist_ok=True)

    # 받은 사진 저장
    for f in student_files + answer_files:
        f.save(os.path.join(upload_folder, f.filename))

    # 파일 처리 함수 호출 (여기에 파일 처리 로직을 작성)
    result_folder = process_result(upload_folder, formatted_exam_date)  # 여기에 result_folder 변수를 반환하도록 수정

    # ZIP 파일 생성
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for root, _, files in os.walk(result_folder):  # result_folder의 파일들을 ZIP으로 압축
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, result_folder))

    zip_buffer.seek(0)

    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f'processed_files_{formatted_exam_date}.zip')

def process_result(upload_folder, formatted_date):
    # result_folder 생성
    result_folder = f'./result_data/{formatted_date}/'
    os.makedirs(result_folder, exist_ok=True)

    # 샘플 엑셀 파일 생성
    excel_file_path = os.path.join(result_folder, 'sample.xlsx')
    df = pd.DataFrame({'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']})
    df.to_excel(excel_file_path, index=False)

    files = os.listdir(upload_folder)

    for f in files:
        img_path = os.path.join(upload_folder, f)
        output_file_path = os.path.join(result_folder, f)
        get_output_img(img_path, output_file_path)

    return result_folder

def reorderPts(pts):
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]  # x좌표로 정렬

    if pts[0, 1] > pts[1, 1]:
        pts[[0, 1]] = pts[[1, 0]]

    if pts[2, 1] < pts[3, 1]:
        pts[[2, 3]] = pts[[3, 2]]

    return pts

def Warping(src_in):
  src_cp = src_in.copy()
  src_copyed = cv2.bitwise_not(src_cp)
  src_gray = cv2.cvtColor(src_cp, cv2.COLOR_BGR2GRAY)
  _, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  contours, _ = cv2.findContours(src_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  h, w = src.shape[:2]
  dw = 500
  dh = round(dw * 297 / 210)  # A4 용지 크기: 210x297cm

  srcQuad = np.array([[30, 30], [30, h - 30], [w - 30, h - 30], [w - 30, 30]], np.float32) # 내가 선택할 모서리 좌표들 ndarray 반시계방향
  dstQuad = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], np.float32)
  dragSrc = [False, False, False, False] # 현재 어떤 점을 드래그 하고 있나의 Flag

  cont = [c for c in contours if (cv2.contourArea(c) > 500 and cv2.contourArea(c) < (h*w-5000))]

  for pts in cont:
      # 외곽선 근사화
      approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True)
      # 컨벡스가 아니고, 사각형이 아니면 무시
      if not cv2.isContourConvex(approx) or len(approx) != 4:
          continue
      srcQuad = reorderPts(approx.reshape(4, 2).astype(np.float32))

  L = (srcQuad[3][1] - srcQuad[0][1])/(srcQuad[3][0] - srcQuad[0][0])
  M = (srcQuad[2][1] - srcQuad[1][1])/(srcQuad[2][0] - srcQuad[1][0])

  ceta = -np.arctan((L+M)/2)

  m_ceta = np.float32( [[ np.cos(ceta), -1* np.sin(ceta), 0.0],
                      [np.sin(ceta), np.cos(ceta), 0.0]])

  r_ceta_INV = cv2.warpAffine(src_copyed,m_ceta,(w,h))

  return r_ceta_INV, srcQuad

def sMaskContur(small_mask_self):
  small_mask_INV_INV = cv2.bitwise_not(small_mask_self)

  ret, otsu = cv2.threshold(small_mask_self, -1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #오츠 알고리즘 이진화
  contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  #윤곽선 검출

  idx = 1
  for cnt in contours:
      if 100000 >cv2.contourArea(cnt) > 30: # 위 사각형 그리기에서 면적 범위를 지정해준 뒤 그 범위 이상일 때만 작동
          x, y, width, height = cv2.boundingRect(cnt) #contours에서 검출된 윤곽선 중 25000의 값을 넘은 선을 감싸는 사각형의 위치와 크기를 얻음
          BBcords.append([x,y,x+width,y+height]) # 좌상단의 x좌표, y좌표, 너비, 높이 순으로 반환된 것을 변환

          cv2.rectangle(std_masked_img_cpy,(x, y),(x + width, y + height), (0, 0, 255), 2) # boundingRect에 저장된 사각형들에 초록색 두께 2인 네모를 그려줌
          idx+=1 # 한 개의 창에 크롭된 이미지를 출력한 후 새로운 창에 출력하기 위해 idx 1증가

  return

def padding_and_shifted (final_image):

  kernel3 = np.ones((2, 2), np.uint8)  # 팽창 및 침식에 사용할 커널
  B_channel = final_image[:, :, 0]
  G_channel = final_image[:, :, 1]
  R_channel = final_image[:, :, 2]

  for _ in range(3):
    if np.any(B_channel > 0):
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)

  M = np.float32([[1, 0, -6], [0, 1, -6]])  # 이동 변환 행렬
  img_shifted = cv2.warpAffine(G_channel, M, (G_channel.shape[1], G_channel.shape[0]))
  modified_image = cv2.merge([B_channel, img_shifted, R_channel])

  return modified_image




def padding_and_shift_binary(binary_image):
    kernel = np.ones((2, 2), np.uint8)  # 팽창 및 침식에 사용할 커널

    for _ in range(3):
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, kernel)

    M = np.float32([[1, 0, -6], [0, 1, -6]])  # 이동 변환 행렬
    img_shifted = cv2.warpAffine(binary_image, M, (binary_image.shape[1], binary_image.shape[0]))

    return img_shifted

def convert_to_3channel(binary_image):
    return cv2.merge([binary_image, binary_image, binary_image])

def apply_otsu_threshold(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image

def binarize_image(image):
    binary_image = np.where(image > 0, 255, 0).astype(np.uint8)
    return binary_image


def shift_image(binary_image):
    M = np.float32([[1, 0, -1], [0, 1, -1]])  # x 방향 -1, y 방향 -1

    h, w = binary_image.shape[:2]

    shifted_image = cv2.warpAffine(binary_image, M, (w, h))

    return shifted_image

def create_inverse_mask(image1, image2):
    if len(image1.shape) == 2:  # 2D 배열 (그레이스케일 이미지)
        image1 = cv2.merge([image1, image1, image1])  # 3채널로 변환

    if len(image2.shape) == 2:  # 2D 배열 (그레이스케일 이미지)
        image2 = cv2.merge([image2, image2, image2])  # 3채널로 변환

    mask1 = np.zeros_like(image1)
    mask2 = np.zeros_like(image2)

    mask1[:, :, 0] = image1[:, :, 0]  # B 채널
    mask1[:, :, 1] = image1[:, :, 1]  # G 채널

    mask2[:, :, 0] = image2[:, :, 0]  # B 채널
    mask2[:, :, 1] = image2[:, :, 1]  # G 채널

    combined_mask = cv2.add(mask1, mask2)

    return combined_mask

def find_connected_components(image):
    num_labels, labels = cv2.connectedComponents(image.astype(np.uint8))
    return labels, num_labels

def calculate_widths(labels, num_labels):
    widths = []
    for i in range(1, num_labels):
        component = (labels == i)
        width = np.max(np.where(component)[1]) - np.min(np.where(component)[1]) + 1
        widths.append(width)
    return np.array(widths)

def calculate_heights(labels, num_labels):
    heights = []
    for i in range(1, num_labels):
        component = (labels == i)
        height = np.max(np.where(component)[0]) - np.min(np.where(component)[0]) + 1
        heights.append(height)
    return np.array(heights)

def label_components_by_width_with_threshold(labels, widths, merge_dist):
    labeled_image = np.zeros_like(labels)
    current_label = 0

    sorted_indices = np.argsort(widths)
    num_components = len(sorted_indices)

    for i in range(num_components):
        index = sorted_indices[i]
        current_width = widths[index]

        if labeled_image[labels == index + 1].any():
            continue  # 이미 라벨이 매겨진 경우 스킵

        current_label += 1
        labeled_image[labels == (index + 1)] = current_label

        for j in range(i + 1, num_components):
            next_index = sorted_indices[j]
            next_width = widths[next_index]

            if abs(next_width - current_width) < merge_dist:
                labeled_image[labels == (next_index + 1)] = current_label
            else:
                break  # 차이가 크면 더 이상 클러스터에 포함되지 않음

    return labeled_image

def label_components_by_height_with_threshold(labels, heights, merge_dist):
    labeled_image = np.zeros_like(labels)
    current_label = 0

    sorted_indices = np.argsort(heights)
    num_components = len(sorted_indices)

    for i in range(num_components):
        index = sorted_indices[i]
        current_height = heights[index]

        if labeled_image[labels == index + 1].any():
            continue  # 이미 라벨이 매겨진 경우 스킵

        current_label += 1
        labeled_image[labels == (index + 1)] = current_label

        for j in range(i + 1, num_components):
            next_index = sorted_indices[j]
            next_height = heights[next_index]

            if abs(next_height - current_height) < merge_dist:
                labeled_image[labels == (next_index + 1)] = current_label
            else:
                break  # 차이가 크면 더 이상 클러스터에 포함되지 않음

    return labeled_image



def visualize_combined_labels(labels_width, labels_height):
    colored_image = np.ones((labels_width.shape[0], labels_width.shape[1], 3), dtype=np.uint8) * 255
    colors = {
        1: (255, 0, 0),  # Red (가로 길이 큰 구성 요소)
        2: (0, 255, 0),  # Green (가로 길이 작은 구성 요소)
        3: (0, 0, 255),  # Blue (세로 길이 큰 구성 요소)
        4: (255, 255, 0) # Yellow (세로 길이 작은 구성 요소)
    }
    for label in range(1, 3):  # 가로 길이에 대한 라벨
        colored_image[labels_width == label] = colors[label]
    for label in range(1, 3):  # 세로 길이에 대한 라벨
        colored_image[labels_height == label] = colors[label]  # 색상 인덱스 조정
    return colored_image

def Change_B_G_R_channel_Seperately(table_in, binary_image_in):
  if len(table_in.shape) == 2:
    table_in = cv2.merge([table_in, table_in, table_in])

  table_in_B = cv2.merge([table_in[:, :, 0], np.zeros_like(table_in[:, :, 0]), np.zeros_like(table_in[:, :, 0])])

  if len(binary_image_in.shape) == 2:
    binary_image_in = cv2.merge([binary_image_in, binary_image_in, binary_image_in])

  org_G_channel_image = cv2.merge([np.zeros_like(binary_image_in[:, :, 0]), binary_image_in[:, :, 1], np.zeros_like(binary_image_in[:, :, 0])])
  final_image = np.zeros_like(table_in)  # 모든 채널이 0으로 초기화된 이미지
  final_image[:, :, 0] = table_in_B[:, :, 0]  # B 채널
  final_image[:, :, 1] = org_G_channel_image[:, :, 1]  # G 채널
  mask = (final_image[:, :, 0] > 0) & (final_image[:, :, 1] > 0)
  final_image[mask] = [0, 0, 0]
  final_image[:, :, 0] = 0
  return final_image

def convert_to_3channel(binary_image):
    return cv2.merge([binary_image, binary_image, binary_image])

def apply_otsu_threshold(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def dilate_and_shift(binary_image):
    kernel = np.ones((2, 2), np.uint8)
    image1 = cv2.dilate(binary_image, kernel)
    image2 = cv2.dilate(image1, kernel)
    eroded_image = image2
    translation_matrix = np.float32([[1, 0, -1], [0, 1, -1]])
    shifted_image = cv2.warpAffine(eroded_image, translation_matrix, (eroded_image.shape[1], eroded_image.shape[0]))
    return shifted_image



def padding_and_shifted (final_image):
  kernel3 = np.ones((2, 2), np.uint8)  # 팽창 및 침식에 사용할 커널
  B_channel = final_image[:, :, 0]
  G_channel = final_image[:, :, 1]
  R_channel = final_image[:, :, 2]

  for _ in range(3):
    if np.any(G_channel > 0):
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_DILATE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)
      G_channel = cv2.morphologyEx(G_channel, cv2.MORPH_ERODE, kernel3)

  M = np.float32([[1, 0, -18], [0, 1, -18]])  # 이동 변환 행렬
  img_shifted = cv2.warpAffine(G_channel, M, (G_channel.shape[1], G_channel.shape[0]))
  modified_image = cv2.merge([B_channel, img_shifted, R_channel])

  return modified_image

def multiply_and_cap_images(img1, img2):
    # 두 이미지가 동일한 크기인지 확인
    if img1.shape != img2.shape:
        raise ValueError("이미지 크기가 일치하지 않습니다.")

    # G 채널 추출
    G_channel_img1 = img1[:, :, 1]  # img1의 G 채널
    G_channel_img2 = img2[:, :, 1]  # img2의 G 채널

    # G 채널 값이 모두 존재하는 경우
    mask_both_exist = (G_channel_img1 > 0) & (G_channel_img2 > 0)

    # 결과 이미지 초기화 (모든 채널이 0)
    result_image = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)

    # G 채널 값으로 설정 (두 값이 모두 존재할 때)
    result_image[mask_both_exist, 1] = G_channel_img2[mask_both_exist]  # G 채널

    return result_image



def combine_images_bitwise_or(img1, img2):

    # 각 채널별로 bitwise OR 연산 수행
    combined_image = np.zeros_like(img1)
    combined_image[:, :, 0] = cv2.bitwise_or(img1[:, :, 0], img2[:, :, 0])  # Blue 채널
    combined_image[:, :, 1] = cv2.bitwise_or(img1[:, :, 1], img2[:, :, 1])  # Green 채널


    return combined_image

# 이미지 path로부터 불러와 src에 복사
img_path = '/content/sample_data/답안지데이터스캔본_215.jpg'
org_image = cv2.imread(img_path)
src = org_image.copy()


# src -> r_ceta_INV # src 파일을 테이블 정렬(r_ceta_INV) 결과 변환
# Warping(src)은 왜곡보정 이미지와 해당 테이블의 좌표를 반환함 -- 투플 형태의 출력
r_ceta_INV, srcQuad_crd = Warping(src)

# r_ceta_INV -> binary_image
# 이미지 불러오기
binary_image = r_ceta_INV


# 1x100, 100x1 크기의 구조 요소 생성
kernel1 = np.ones((1, 100), np.uint8)
kernel2 = np.ones((100, 1), np.uint8)
kernel3 = np.ones((2, 2), np.uint8)  # 팽창 및 침식에 사용할 커널


# 모폴로지 침식/팽창 연산 적용
eroded_image1 = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, kernel1) # binary로부터 1x100 커널요소 침식 연산
eroded_image2 = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, kernel2) # binary로부터 100x1 커널요소 침식 연산

dilate_image3 = cv2.morphologyEx(eroded_image1, cv2.MORPH_DILATE, kernel1) # 1x100 커널 침식에 1x100 커널 팽창 (가로성분)
dilate_image4 = cv2.morphologyEx(eroded_image2, cv2.MORPH_DILATE, kernel2) # 100x1 커널 침식에 100x1 커널 팽창 (세로성분)

eroded_image0 = cv2.bitwise_or(dilate_image3, dilate_image4) # 두 테이블 요소 합치기



################################################################################


# saturate_r_ceta : r_ceta_copy 이미지에 대한 왼쪽으로 1픽셀, 위로 1픽셀 이동한 이미지
r_ceta_copy = r_ceta_INV.copy()
saturate_r_ceta = shift_image(r_ceta_copy)

# dilate_image3 : 가로성분
# dilate_image3_saturated : 가로성분의 padding&shifting (3 채널)
dilate_image3_saturated = padding_and_shift_binary(dilate_image3)
# dilate_image3_saturated_bin : 가로성분의 padding&shifting (binary)
dilate_image3_saturated_bin = apply_otsu_threshold(dilate_image3_saturated)

# dilate_image4 : 세로성분
# dilate_image4_saturated : 세로성분의 padding&shifting (3 채널)
# dilate_image4_saturated_bin : 세로성분의 padding&shifting (binary)
dilate_image4_saturated = padding_and_shift_binary(dilate_image4)
dilate_image4_saturated_bin = apply_otsu_threshold(dilate_image4_saturated)

# eroded_image0 : 가로성분 + 세로성분 (3채널)
# eroded_image0_saturated : 테이블 성분 padding&shifting (3채널)
# eroded_image0_saturated_bin : 테이블 성분 padding&shifting (binary)
#eroded_image0_saturated = padding_and_shift_binary(eroded_image0)
#eroded_image0_saturated_bin = apply_otsu_threshold(eroded_image0_saturated)
eroded_image0_bin = apply_otsu_threshold(eroded_image0)



################################################################################################################

# eroded_image0_saturated_bin : 테이블 성분 padding&shifting (binary)
# table_in : 테이블 성분 padding&shifting (binary)

table_in = eroded_image0_bin
table_in_saturated = dilate_and_shift(table_in)
table_in_saturated_copy = table_in_saturated.copy()
binary_image_in = binary_image.copy()
# cv2_imshow(binary_image_in)

final_image_out = Change_B_G_R_channel_Seperately(table_in_saturated_copy, binary_image_in)
final_image_out[final_image_out[:, :, 0] > 0] = 0
final_out_result = final_image_out
# cv2_imshow(final_out_result)  # B채널 지운 G채널 이미지 표시


# B 채널과 G 채널 분리한 마스크들을 합쳐서 표시 (확인용)
#final_image_merged = create_inverse_mask(table_in_saturated, final_image_out)
#cv2_imshow(final_image_merged)


# binary_image를 3채널 이미지로 변환
final_image_out_copy = final_out_result.copy()
final_image_bin = apply_otsu_threshold(final_image_out_copy)
three_channel_image = convert_to_3channel(final_image_bin)
# cv2_imshow(three_channel_image)
img_copy = three_channel_image.copy()
img_copy_INV = cv2.bitwise_not(img_copy)


################################################################################
# 태이블 계산
################################################################################

# 연결된 구성 요소 찾기
labels_row, num_labels_row = find_connected_components(dilate_image3_saturated_bin)
labels_col, num_labels_col = find_connected_components(dilate_image4_saturated_bin)

# 각 클러스터의 가로 및 세로 픽셀 수 계산
widths = calculate_widths(labels_row, num_labels_row)
heights = calculate_heights(labels_col, num_labels_col)

# 라벨링
merge_dist = 10  # 예시 값, 원하는 거리로 설정
width_labeled_image = label_components_by_width_with_threshold(labels_row, widths, merge_dist)
height_labeled_image = label_components_by_height_with_threshold(labels_col, heights, merge_dist)

# 라벨 시각화
visualized_combined_image = visualize_combined_labels(width_labeled_image, height_labeled_image)

# 결과 시각화
# cv2_imshow(visualized_combined_image)
#cv2_imshow(dilate_image3_saturated_bin)
#cv2_imshow(dilate_image4_saturated_bin)






def find_connected_components_row(image):
    num_labels, labels = cv2.connectedComponents(image.astype(np.uint8))
    components = []
    widths_centers = []

    for i in range(1, num_labels):
        component = (labels == i)
        components.append(component)

        # 가로 픽셀 수 계산
        width = np.max(np.where(component)[1]) - np.min(np.where(component)[1]) + 1

        # 중심 좌표 계산
        y_coords, x_coords = np.where(component)
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        widths_centers.append((width, (center_x, center_y)))

    return labels, num_labels, components, widths_centers

def find_connected_components_col(image):
    num_labels, labels = cv2.connectedComponents(image.astype(np.uint8))
    components = []
    heights_centers = []

    for i in range(1, num_labels):
        component = (labels == i)
        components.append(component)

        # 세로 픽셀 수 계산
        height = np.max(np.where(component)[0]) - np.min(np.where(component)[0]) + 1

        # 중심 좌표 계산
        y_coords, x_coords = np.where(component)
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        heights_centers.append((height, (center_x, center_y)))

    return labels, num_labels, components, heights_centers

def label_by_widths(widths_centers, merge_dist):
    sorted_indices = np.argsort([wc[0] for wc in widths_centers])
    labels = np.zeros(len(widths_centers), dtype=int)
    current_label = 0

    for i in range(len(sorted_indices)):
        if labels[sorted_indices[i]] == 0:
            current_label += 1
            labels[sorted_indices[i]] = current_label

            for j in range(i + 1, len(sorted_indices)):
                if abs(widths_centers[sorted_indices[j]][0] - widths_centers[sorted_indices[i]][0]) < merge_dist:
                    labels[sorted_indices[j]] = current_label

    labeled_components_width = [(label, widths_centers[i]) for i, label in enumerate(labels)]
    return labeled_components_width

def label_by_heights(heights_centers, merge_dist):
    sorted_indices = np.argsort([hc[0] for hc in heights_centers])
    labels = np.zeros(len(heights_centers), dtype=int)
    current_label = 0

    for i in range(len(sorted_indices)):
        if labels[sorted_indices[i]] == 0:
            current_label += 1
            labels[sorted_indices[i]] = current_label

            for j in range(i + 1, len(sorted_indices)):
                if abs(heights_centers[sorted_indices[j]][0] - heights_centers[sorted_indices[i]][0]) < merge_dist:
                    labels[sorted_indices[j]] = current_label

    labeled_components_height = [(label, heights_centers[i]) for i, label in enumerate(labels)]
    return labeled_components_height



def get_labeled_list(widths_centers_in, heights_centers_in, merge_dist_in=10):
  labeled_components_width = label_by_widths(widths_centers_in, merge_dist_in)
  labeled_components_height = label_by_heights(heights_centers_in, merge_dist_in)

  # 같은 라벨에 속하는 labeled_components_width의 y좌표들과 labeled_components_height의 x좌표들 끼리의 집합으로 labeled_list 계산
  labeled_list = []
  for w_label, (width, (cx, cy)) in labeled_components_width:
      for h_label, (height, (hx, hy)) in labeled_components_height:
          if w_label == h_label:  # 같은 라벨
              labeled_list.append((w_label, [(hx, cy)]))
  return labeled_list






# def get_max_row_col_index(list_in):

# labeled_list

def get_bounding_box_coords(label, cel_row, cel_col):
    xi, yi = sorted_array_rowcol[label][cel_col][cel_row][1]

    # (x_{i+1}, y_{i+1}) - next coordinate (bounding box right or bottom)
    if (cel_row + 1 < len(sorted_array_rowcol[label][cel_col])) and (cel_col + 1 < len(sorted_array_rowcol[label])):
        # If both row and column have next entries
        x_next, y_next = sorted_array_rowcol[label][cel_col + 1][cel_row + 1][1]
    elif cel_row + 1 > len(sorted_array_rowcol[label][cel_col]) and cel_col + 1 < len(sorted_array_rowcol[label]):
        # If row exceeds but column has next
        x_next, y_next = sorted_array_rowcol[label][cel_col + 1][0][1]
    elif cel_col + 1 > len(sorted_array_rowcol[label]) and cel_row + 1 < len(sorted_array_rowcol[label][cel_col]):
        # If column exceeds but row has next
        x_next, y_next = sorted_array_rowcol[label][cel_col][cel_row + 1][1]
    # elif ((cel_col + 1 == len(sorted_array_rowcol[label])) and (cel_row + 1 == len(sorted_array_rowcol[label][cel_col]))):
        # If column exceeds but row has next
        # x_next, y_next = sorted_array_rowcol[label][cel_col][cel_row][0]
    else:
        # Last element
        x_next, y_next = xi, yi

    # Return cel_row, cel_col along with bounding box coordinates
    return (cel_row, cel_col, xi, yi, x_next, y_next)






# 예시로 사용하기 위한 코드
input_image_width = dilate_image3_saturated_bin  # 입력 이미지1
input_image_height = dilate_image4_saturated_bin  # 입력 이미지2

labels_row, num_labels_row, components_row, widths_centers = find_connected_components_row(input_image_width)
labels_col, num_labels_col, components_col, heights_centers = find_connected_components_col(input_image_height)

labeled_list = get_labeled_list(widths_centers, heights_centers, 10)


# array[label] 형태로 변환할 defaultdict 생성
labeld_array = defaultdict(list)



# labeled_list에서 라벨별로 좌표들을 labeld_array[label]로 저장
for label, points in labeled_list:
    for (x, y) in points:
        labeld_array[label].append((x, y))



# labeld_array의 각 라벨에 대해 (x, y) 좌표를 먼저 x에 대해 정렬하고, 동일한 x에 대해서는 y에 대해 정렬
for label in labeld_array:
    labeld_array[label] = sorted(labeld_array[label], key=lambda point: (point[0], point[1]))

# sorted_array_rowcol 배열 생성
sorted_array_rowcol = defaultdict(lambda: defaultdict(list))



max_coords = {}

for label, points in labeld_array.items():
    # x와 y 좌표별 최대값을 구합니다
    max_x = max(points, key=lambda p: p[0])[0]  # x값의 최대값
    max_y = max(points, key=lambda p: p[1])[1]  # y값의 최대값
    max_coords[label] = (max_x, max_y)


# 각 라벨의 좌표에 대해 col_index와 row_index를 할당
for label, points in labeld_array.items():
    col_index = 0
    for x in sorted(set(point[0] for point in points)):  # x 값에 대해 순서대로 col_index 할당

        row_points = sorted([point for point in points if point[0] == x], key=lambda point: point[1])
        row_index = 0
        for (x, y) in row_points:
            sorted_array_rowcol[label][col_index].append((row_index, (x, y)))
            row_index += 1
        col_index += 1

# 2. Generate bounding boxes for visualization (cel_row, cel_col, min_x, min_y, max_x, max_y)
bounding_boxes = {}



# for label in labeld_array:
#     for col_index in sorted_array_rowcol[label]:
#         for row_index, (x, y) in sorted_array_rowcol[label][col_index]:
#             # For each label, extract coordinates and create bounding box
#             cel_row, cel_col, min_x, min_y, max_x, max_y = get_bounding_box_coords(label, row_index, col_index)
#             if label not in bounding_boxes:
#                 bounding_boxes[label] = []
#             bounding_boxes[label].append((cel_row, cel_col, min_x, min_y, max_x, max_y))
#
# print(bounding_boxes)

for label in labeld_array:
    for col_index in sorted_array_rowcol[label]:
        for row_index, (x, y) in sorted_array_rowcol[label][col_index]:
            # For each label, extract coordinates and create bounding box
            cel_row, cel_col, min_x, min_y, max_x, max_y = get_bounding_box_coords(label, row_index, col_index)
            if label not in bounding_boxes:
                bounding_boxes[label] = []
            bounding_boxes[label].append((cel_row, cel_col, min_x, min_y, max_x, max_y))

# print(bounding_boxes)


# bounding_boxes에서 각 label별로 cel_row와 cel_col의 최댓값을 찾기
max_values = {}

for label, bbox_list in bounding_boxes.items():
    max_cel_row = max_cel_col = -float('inf')  # 초기값을 아주 작은 값으로 설정
    for (cel_row, cel_col, min_x, min_y, max_x, max_y) in bbox_list:
        max_cel_row = max(max_cel_row, cel_row)  # cel_row의 최댓값을 찾음
        max_cel_col = max(max_cel_col, cel_col)  # cel_col의 최댓값을 찾음

    # 각 label별로 cel_row, cel_col의 최댓값을 저장
    max_values[label] = (max_cel_row, max_cel_col)


# 기존 bounding_boxes에서 max_values의 최댓값을 제외하기
for label, bbox_list in bounding_boxes.items():
    if label in max_values:
        max_cel_row, max_cel_col = max_values[label]
        # bounding_boxes의 bbox_list에서 cel_row와 cel_col이 max값인 항목을 제외
        filtered_bbox_list = [
            (cel_row, cel_col, min_x, min_y, max_x, max_y)
            for (cel_row, cel_col, min_x, min_y, max_x, max_y) in bbox_list
            if not (cel_row == max_cel_row or cel_col == max_cel_col)  # max값인 항목을 제외
        ]
        # filtered_bbox_list로 bounding_boxes 업데이트
        bounding_boxes[label] = filtered_bbox_list

print(bounding_boxes)

ordered_list = {}


# 기존 bounding_boxes에서 max_values의 최댓값을 제외하기
for label, bbox_list in bounding_boxes.items():
    if label in max_values:
        max_cel_row, max_cel_col = max_values[label]
        # bounding_boxes의 bbox_list에서 cel_row와 cel_col이 max값인 항목을 제외
        filtered_bbox_list = [
            (cel_row, cel_col, min_x, min_y, max_x, max_y)
            for (cel_row, cel_col, min_x, min_y, max_x, max_y) in bbox_list
            if not (cel_row == max_cel_row or cel_col == max_cel_col)  # max값인 항목을 제외
        ]
        for cel_row in range(max_cel_row-1):
          ordered_list[label][cel_row] = [
            (cel_col, min_x, min_y, max_x, max_y)]
        for cel_row in range(cel_row-1):
               # (#sorted_list[])



        # filtered_bbox_list로 bounding_boxes 업데이트
        bounding_boxes[label] = filtered_bbox_list



print(bounding_boxes)





# 3. Generate colors for bounding boxes
def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        color = [np.random.randint(0, 256) for _ in range(3)]
        colors.append(color)
    return colors

colors = generate_colors(len(bounding_boxes))

def visualize_bounding_boxes(image, bounding_boxes, colors):
    image_copy = image.copy()

    # Iterate over each label and its corresponding bounding boxes
    for label, boxes in bounding_boxes.items():
        color = colors[label % len(colors)]  # Select color for the label
        for (cel_row, cel_col, min_x, min_y, max_x, max_y) in boxes:
            # Draw the rectangle
            cv2.rectangle(image_copy, (min_x, min_y), (max_x, max_y), color, 2)

            # Set the position for the text (Cel Row and Cel Column)
            text_position = ((min_x + (max_x - min_x) // 2) - 2, min_y + (max_y - min_y) // 2)  # Center of the box

            # Prepare the text to display
            text = f"({cel_row},{cel_col})"

            # Add a background rectangle for the text (black background for better visibility)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_width, text_height = text_size
            cv2.rectangle(image_copy, (text_position[0] - 15, text_position[1] - 15),
                          (text_position[0] + text_width + 7, text_position[1] + text_height + 7), (0, 0, 0), -1)

            # Add the text on top of the rectangle
            cv2.putText(image_copy, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return image_copy

# 입력 이미지1, 이미지2를 bitwise_and 시킨 영상
input_image_width = dilate_image3_saturated_bin  # 입력 이미지
input_image_height = dilate_image4_saturated_bin  # 입력 이미지
combined_image = cv2.bitwise_and(input_image_width, input_image_height)
combined_image_INV = cv2.bitwise_not(combined_image)


# Assuming combined_image_INV is an existing image (gray scale)
combined_image_INV_3chan = cv2.cvtColor(combined_image_INV, cv2.COLOR_GRAY2BGR)

# Resulting image with bounding boxes
bounding_img = visualize_bounding_boxes(combined_image_INV_3chan, bounding_boxes, colors)

# Display the image (using OpenCV)
# cv2_imshow(combined_image_INV_3chan)
cv2_imshow(bounding_img)



# print(bounding_boxes)
# {2: [(0, 0, 161, 356, 275, 385), (1, 0, 161, 385, 275, 452), (2, 0, 161, 452, 275, 520), (3, 0, 161, 520, 275, 587), (4, 0, 161, 587, 275, 654), (5, 0, 161, 654, 275, 721), (6, 0, 161, 721, 275, 788), (7, 0, 161, 788, 275, 855), (8, 0, 161, 855, 275, 922), (9, 0, 161, 922, 275, 989), (10, 0, 161, 989, 275, 1135), (11, 0, 161, 1135, 275, 1262), (12, 0, 161, 1262, 275, 1322), (13, 0, 161, 1322, 275, 1382), (14, 0, 161, 1382, 275, 1441), (15, 0, 161, 1441, 275, 1500), (16, 0, 161, 1500, 275, 1560), (0, 1, 275, 356, 361, 385), (1, 1, 275, 385, 361, 452), (2, 1, 275, 452, 361, 520), (3, 1, 275, 520, 361, 587), (4, 1, 275, 587, 361, 654), (5, 1, 275, 654, 361, 721), (6, 1, 275, 721, 361, 788), (7, 1, 275, 788, 361, 855), (8, 1, 275, 855, 361, 922), (9, 1, 275, 922, 361, 989), (10, 1, 275, 989, 361, 1135), (11, 1, 275, 1135, 361, 1262), (12, 1, 275, 1262, 361, 1322), (13, 1, 275, 1322, 361, 1382), (14, 1, 275, 1382, 361, 1441), (15, 1, 275, 1441, 361, 1500), (16, 1, 275, 1500, 361, 1560), (0, 2, 361, 356, 1043, 385), (1, 2, 361, 385, 1043, 452), (2, 2, 361, 452, 1043, 520), (3, 2, 361, 520, 1043, 587), (4, 2, 361, 587, 1043, 654), (5, 2, 361, 654, 1043, 721), (6, 2, 361, 721, 1043, 788), (7, 2, 361, 788, 1043, 855), (8, 2, 361, 855, 1043, 922), (9, 2, 361, 922, 1043, 989), (10, 2, 361, 989, 1043, 1135), (11, 2, 361, 1135, 1043, 1262), (12, 2, 361, 1262, 1043, 1322), (13, 2, 361, 1322, 1043, 1382), (14, 2, 361, 1382, 1043, 1441), (15, 2, 361, 1441, 1043, 1500), (16, 2, 361, 1500, 1043, 1560)], 1: [(0, 0, 474, 998, 540, 1025), (1, 0, 474, 1025, 540, 1052), (2, 0, 474, 1052, 540, 1079), (3, 0, 474, 1079, 540, 1105), (0, 1, 540, 998, 607, 1025), (1, 1, 540, 1025, 607, 1052), (2, 1, 540, 1052, 607, 1079), (3, 1, 540, 1079, 607, 1105), (0, 2, 607, 998, 674, 1025), (1, 2, 607, 1025, 674, 1052), (2, 2, 607, 1052, 674, 1079), (3, 2, 607, 1079, 674, 1105), (0, 3, 674, 998, 741, 1025), (1, 3, 674, 1025, 741, 1052), (2, 3, 674, 1052, 741, 1079), (3, 3, 674, 1079, 741, 1105)]}





## 기존 bounding_boxes에서 max_values의 최댓값을 제외하기
#for label, bbox_list in bounding_boxes.items():
#    if label in max_values:
#        max_cel_row, max_cel_col = max_values[label]
#        # bounding_boxes의 bbox_list에서 cel_row와 cel_col이 max값인 항목을 제외
#        filtered_bbox_list = [
#            (cel_row, cel_col, min_x, min_y, max_x, max_y)
#            for (cel_row, cel_col, min_x, min_y, max_x, max_y) in bbox_list
#            if not (cel_row == max_cel_row or cel_col == max_cel_col)  # max값인 항목을 제외
#        ]
#        # filtered_bbox_list로 bounding_boxes 업데이트
#        bounding_boxes[label] = filtered_bbox_list

#print(bounding_boxes)



# Step 1: 최댓값 제외하기
for label, bbox_list in bounding_boxes.items():
    if label in max_values:
        max_cel_row, max_cel_col = max_values[label]
        # bounding_boxes의 bbox_list에서 cel_row와 cel_col이 max값인 항목을 제외
        filtered_bbox_list = [
            (cel_row, cel_col, min_x, min_y, max_x, max_y)
            for (cel_row, cel_col, min_x, min_y, max_x, max_y) in bbox_list
            if not (cel_row == max_cel_row or cel_col == max_cel_col)  # max값인 항목을 제외
        ]
        # filtered_bbox_list로 bounding_boxes 업데이트
        bounding_boxes[label] = filtered_bbox_list

print(max_cel_row, max_cel_col)

ordered_list = {}





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
