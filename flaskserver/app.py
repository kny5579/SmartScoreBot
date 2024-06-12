from flask import Flask, render_template, request, send_file, url_for
import zipfile
import os
from io import BytesIO
from PIL import Image
import pandas as pd
from datetime import datetime, timezone

# import the necessary packages for OCR

import pytesseract
from  sklearn.cluster  import  AgglomerativeClustering
from  pytesseract import Output

import pytesseract
from PIL import ImageEnhance, ImageFilter, Image
pytesseract.pytesseract.tesseract_cmd = ( r'/usr/bin/tesseract' )

# import  pandas  as  pd
import numpy as np
import pytesseract
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from skimage import data
from skimage.filters import threshold_otsu, threshold_local
# import os
import glob





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

    # 파일 처리 함수 호출
    result_path = process_result(upload_folder)

    # ZIP 파일 생성
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for root, _, files in os.walk(result_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, result_path))

    zip_buffer.seek(0)

    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f'processed_files_{formatted_date}.zip')


def process_result(upload_folder):
    # result_folder 생성
    result_path = os.path.join(upload_folder, "result_file")
    os.makedirs(result_path, exist_ok=True)

    # 샘플 엑셀 파일 생성
    excel_file_path = os.path.join(result_path, 'sample.xlsx')
    df = pd.DataFrame({'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']})
    df.to_excel(excel_file_path, index=False)

    # 샘플 이미지 파일 생성 (이 부분은 주석 처리되어 있으므로 예시로 남겨 둠)
    # image_file_path = os.path.join(result_path, 'sample.jpg')
    # 이미지 파일 생성 로직 추가

    for f in os.listdir(upload_folder):
        img_path = os.path.join(upload_folder, f)
        output_file_path = os.path.join(result_path, f)
        if os.path.isfile(img_path):  # 파일인지 확인
            get_output_img(img_path, output_file_path)

    return result_path





def get_output_img(img_path, output_file_path):
    COL = 3
    ROW = 17
    Mergdist = 10 # 동일 line의 픽셀에 해당하는 margin distance
    ANSWER_ROW = 3 # 답안 열 시작부분
    ANSWER_COL = 2 # 답안 행 시작 부분


    x_coords = [] # answer page에 찾은 contour들의 x좌표 리스트
    y_coords = [] ## answer page에 찾은 contour들의 x좌표 리스트y좌표 리스트
    BBcords = []  # answer page에 찾은 contour들의 x좌표 리스트 boundingbox 좌표[x,y,x+w, y+h]
    TbBBcords = []  # table page에 찾은 contour들의 x좌표 리스트 boundingbox 좌표[x,y,x+w, y+h]
    Tb_cords_x = []
    Tb_cords_y = []
    cont_Table = []


    try:
        org_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Warping(src)은 왜곡보정 이미지와 해당 테이블의 좌표를 반환함 -- 투플 형태의 출력
        src = org_image.copy()
        r_ceta_INV, srcQuad_crd = Warping(org_image)

        # srcQuad_crd을 이용해 student table masking 작업

        # WarpedImg는 Inversed Image이므로 해당 이미지를 다시 흰색으로 만들어 사용함 검은색 line을 그린 이미지는 color channel값이 들어간 이미지이므로 copy 및 binarization을 다시 해야 함
        r_ceta_INV_INV = cv2.bitwise_not(r_ceta_INV)

        # WeakLinemaskImg : INV, copyed_img
        WeakLineMaskImg_INV = draw_WeakLineMask(r_ceta_INV_INV)


        # Hard line masking (White->Black)
        HardLineMaskedIMG = draw_HardLineMask(r_ceta_INV)
        # r_ceta_INV
        # r_input이 흰색인지 명확하지 않음
        # draw_HardLineMask(r_ceta_INV_INV)
        # Black Masking
        # output : [HLM_INV_OUT]
        # def draw_HardLineMask(r_input):

        # Hard line masking에 대한 thin kernel masking
        masking_for_std = get_small_mask(HardLineMaskedIMG)

        # Table을 거의 제거한 std_img
        # INV->INV
        WeakLineMaskImg_INV_INV = cv2.bitwise_not(WeakLineMaskImg_INV)
        std_masked_img = Masking_src(masking_for_std, WeakLineMaskImg_INV)

        # TesseractOCR은 흰색이 더 성능이 나음
        # LineMaskedImg_INV = cv2.bitwise_not(LineMaskedImg).copy()

        # thinMasking 이미지에 대한 findContour


        std_masked_img_cpy = WeakLineMaskImg_INV_INV.copy()
        #print(" ")
        #print("sMaskContur(masking_for_std)")


        std_masked_img_cpy_Draw_contour = sMaskContur(masking_for_std, std_masked_img_cpy)


        # r_ceta_INV_INV_copyINV //
        #print("r_ceta_INV_INV_copyINV")
        r_ceta_INV_INV_copyINV = r_ceta_INV_INV.copy()
        #cv2_imshow(r_ceta_INV_INV_copyINV)

        # HardLineMaskedIMG_copy
        #print("HardLineMaskedIMG_copy")
        HardLineMaskedIMG_copy = HardLineMaskedIMG.copy()
        #cv2_imshow(HardLineMaskedIMG_copy)

        #print("TbMask_Stu_img")
        TbMask_Stu_img = cv2.bitwise_or(r_ceta_INV_INV_copyINV, HardLineMaskedIMG_copy)
        #cv2_imshow(TbMask_Stu_img)

        # 최종적으로 찾은 bounding box
        #cv2_imshow(std_masked_img_cpy)
        output_file_path = os.path.join(output_file_path, "output_img.jpg")
        cv2.imwrite(output_file_path, std_masked_img_cpy_Draw_contour)

    except AttributeError as A:
        print('AttributeError: NoneType object has no attribute copy', A)

    return

# 사용자 정의 함수

new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,  # np.float128 ; doesn't exist on windows
    'G': np.complex128,  # np.complex256 ; doesn't exist on windows
}



def convert_to_float(image, preserve_range):
    """Convert input image to float image with the appropriate range.

    Parameters
    ----------
    image : ndarray
      Input image.
    preserve_range : bool
      Determines if the range of the image should be kept or transformed
      using img_as_float. Also see
      https://scikit-image.org/docs/dev/user_guide/data_types.html

    Notes
    -----
    * Input images with `float32` data type are not upcast.

    Returns
    -------
    image : ndarray
      Transformed version of the input.

    """
    if image.dtype == np.float16:
        return image.astype(np.float32)
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        from ..util.dtype import img_as_float

        image = img_as_float(image)
    return image




def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect



def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped




def plt_imshow(title='image', img=None, figsize=(40 ,30)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()



def reorderPts(pts):
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]

    if pts[0, 1] > pts[1, 1]:
        pts[[0, 1]] = pts[[1, 0]]

    if pts[2, 1] < pts[3, 1]:
        pts[[2, 3]] = pts[[3, 2]]

    return pts




COL = 3
ROW = 17
Mergdist = 10 # 동일 line의 픽셀에 해당하는 margin distance
ANSWER_ROW = 3 # 답안 열 시작부분
ANSWER_COL = 2 # 답안 행 시작 부분


x_coords = [] # answer page에 찾은 contour들의 x좌표 리스트
y_coords = [] ## answer page에 찾은 contour들의 x좌표 리스트y좌표 리스트
BBcords = []  # answer page에 찾은 contour들의 x좌표 리스트 boundingbox 좌표[x,y,x+w, y+h]
TbBBcords = []  # table page에 찾은 contour들의 x좌표 리스트 boundingbox 좌표[x,y,x+w, y+h]
Tb_cords_x = []
Tb_cords_y = []
cont_Table = []




def reorderPts(pts):
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]  # x좌표로 정렬

    if pts[0, 1] > pts[1, 1]:
        pts[[0, 1]] = pts[[1, 0]]

    if pts[2, 1] < pts[3, 1]:
        pts[[2, 3]] = pts[[3, 2]]

    return pts

# ----------------------------------------------------------------------
# def Warping(src_in): // src_in 이미지를 받아 내부적 복사를 한 뒤 회전 보정
#  return r_ceta_INV, srcQuad // Inversed 회전보정 이미지와 테이블 좌표 리스트 srcQuad 반환
# ----------------------------------------------------------------------



def Warping(src_in):

    src_cp = src_in.copy()
    src_copyed = cv2.bitwise_not(src_cp)
    src_gray = cv2.cvtColor(src_cp, cv2.COLOR_BGR2GRAY)
    _, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(src_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)



    h, w = src_cp.shape[:2]
    dw = 500
    dh = round(dw * 297 / 210)  # A4 용지 크기: 210x297cm



    # 좌표 변수 등
    # 모서리 점들의 좌표, 드래그 상태 여부
    # 위에서 이미 정의한 함수에서 글로벌 변수로 사용되었던 것들의 존재가 바로 이곳에 있다.

    srcQuad = np.array([[30, 30], [30, h - 30], [w - 30, h - 30], [w - 30, 30]], np.float32) # 내가 선택할 모서리 좌표들 ndarray 반시계방향
    dstQuad = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], np.float32)
    dragSrc = [False, False, False, False] # 현재 어떤 점을 드래그 하고 있나의 Flag


    # 자잘한 도형 거르기
    # 사실 영상 속에서 도형과 외곽선이라는 것이 하나둘만 생기는 게 아니라 수없이 많다.
    # 그래서 대강 contourArea가 1000을 넘는, 즉 어느 정도 크기를 가지는 도형들에 대해서만 처리를 하기 위해 미리 한 번 걸렀다.
    # 이번 영상의 경우 contourArea가 1119.5인 도형과 232636.5인 도형 딱 두 개만 살아남고 나머지는 다 걸러진다.

    cont = [c for c in contours if (cv2.contourArea(c) > 500 and cv2.contourArea(c) < (h*w-5000))]


    # 사각형 검출 및 좌표 재지정
    # 위에서 한 번 거른 도형 리스트를 for문을 돌린다.
    # 여기에서는 살아남았던 도형 중에서도 사각형이 아니면(A4용지 전제이므로) 또는 contour convex라면(오목한 부분이 있는지 - 종이라면 있을 리가 없기 때문에) skip하고, 아니면 계속 이어나간다.
    # 여기까지 살아남은 contour라면 좌표를 재지정한다.

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
    # cv2_imshow(r_ceta_INV)

    return r_ceta_INV, srcQuad


def draw_WeakLineMask (r_input):
    src_r_input = r_input.copy()
    gray = cv2.cvtColor(src_r_input, cv2.COLOR_BGR2GRAY) #흑백화
    ret, otsu = cv2.threshold(gray, -1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #오츠 알고리즘 이진화

    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  #윤곽선 검출

    COLOR = (255,255,255)


    idx = 1; # crop된 이미지가 서로 다른 창에 뜰 수 있도록 idx를 사용
    for cnt in contours:
        if 100000 >cv2.contourArea(cnt) > 1000: # 위 사각형 그리기에서 면적 범위를 지정해준 뒤 그 범위 이상일 때만 작동
            x, y, width, height = cv2.boundingRect(cnt) #contours에서 검출된 윤곽선 중 25000의 값을 넘은 선을 감싸는 사각형의 위치와 크기를 얻음
            TbBBcords.append([x,y,x+width,y+height]) # 좌상단의 x좌표, y좌표, 너비, 높이 순으로 반환된 것을 변환
            x_coords.append(x)
            x_coords.append(x)
            x_coords.append(x+width)
            x_coords.append(x+width)
            y_coords.append(y)
            y_coords.append(y+height)
            y_coords.append(y)
            y_coords.append(y+height)

            cv2.rectangle(src_r_input,(x, y),(x + width, y + height), COLOR, 1) # boundingRect에 저장된 사각형들에 초록색 두께 2인 네모를 그려줌
            idx+=1 # 한 개의 창에 크롭된 이미지를 출력한 후 새로운 창에 출력하기 위해 idx 1증가

    idx = 1
    for cnt in contours:
        if 100000 >cv2.contourArea(cnt) > 1000: # 위 사각형 그리기에서 면적 범위를 지정해준 뒤 그 범위 이상일 때만 작동
            cont_Table.append(cnt)
            cv2.rectangle(src_r_input,(x, y),(x + width, y + height), COLOR, 1) # boundingRect에 저장된 사각형들에 초록색 두께 2인 네모를 그려줌
            idx+=1 # 한 개의 창에 크롭된 이미지를 출력한 후 새로운 창에 출력하기 위해 idx 1증가


    # print(BBcords)
    #        crop = r_ceta_copy[y:y+height, x:x+width] #위 작업으로 얻은 사각형의 위치를 원본 이미지에서 잘라옴
    #dst = cv2.resize(crop,None,fx=1.5,fy=1.5) #크롭된 이미지의 크기가 너무 작아 크기 조정해줌
    #        cv2_imshow(crop) #원본에서 잘라온 이미지를 출력
    #cv2.imwrite(f'crop_{idx}.png',dst) # 저장

    # cv2_imshow(src_r_input)
    # cv2.imwrite('/content/sample_data/output.jpg', r_ceta_copy)
    cvtBin_Img = src_r_input.copy()
    cvtBin_Img_out = cv2.bitwise_not(cvtBin_Img)
    return cvtBin_Img_out


# cv2_imshow(WeakLineMaskImg_INV)







# ------------------------------------------------------------------------------------------------
# def draw_HardLineMask (r_input): // rotate 보정된 원본 이미지를 내부적으로 흑백으로 복사하여 그래프 그림
#   return HLM_INV_OUT // Hard Line이 그려진 흑백 이미지를 출력
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# 전역
# x_coords = []
# y_coords = []
# BBcords = []
# Tb_cords = []
# ------------------------------------------------------------------------------------------------



def draw_HardLineMask(r_input):
    r_input_copy = r_input.copy()
    SrcImg_HLM_INV = cv2.bitwise_not(r_input_copy)
    gray = cv2.cvtColor(SrcImg_HLM_INV, cv2.COLOR_BGR2GRAY) #흑백화
    ret, otsu = cv2.threshold(gray, -1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #오츠 알고리즘 이진화

    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  #윤곽선 검출


    idx = 1; # crop된 이미지가 서로 다른 창에 뜰 수 있도록 idx를 사용
    for cnt in contours:
        if 100000 >cv2.contourArea(cnt) > 1000: # 위 사각형 그리기에서 면적 범위를 지정해준 뒤 그 범위 이상일 때만 작동
            x, y, width, height = cv2.boundingRect(cnt) #contours에서 검출된 윤곽선 중 25000의 값을 넘은 선을 감싸는 사각형의 위치와 크기를 얻음
            # BBcords.append([x,y,x+width,y+height]) # 좌상단의 x좌표, y좌표, 너비, 높이 순으로 반환된 것을 변환
            x_coords.append(x)
            x_coords.append(x)
            x_coords.append(x+width)
            x_coords.append(x+width)
            y_coords.append(y)
            y_coords.append(y+height)
            y_coords.append(y)
            y_coords.append(y+height)

            cv2.rectangle(r_input_copy,(x, y),(x + width, y + height), (0,0,0), 3) # boundingRect에 저장된 사각형들에 초록색 두께 2인 네모를 그려줌
            idx+=1 # 한 개의 창에 크롭된 이미지를 출력한 후 새로운 창에 출력하기 위해 idx 1증가

    idx = 1
    for cnt in contours:
        if 100000 >cv2.contourArea(cnt) > 1000: # 위 사각형 그리기에서 면적 범위를 지정해준 뒤 그 범위 이상일 때만 작동
            cv2.rectangle(r_input_copy,(x, y),(x + width, y + height), (0,0,0), 3) # boundingRect에 저장된 사각형들에 초록색 두께 2인 네모를 그려줌
            idx+=1 # 한 개의 창에 크롭된 이미지를 출력한 후 새로운 창에 출력하기 위해 idx 1증가

    # gray_out = cv2.cvtColor(r_input_copy, cv2.COLOR_BGR2GRAY)
    # HLM_INV_OUT = cv2.bitwise_not(gray_out)
    HLM_INV_OUT = r_input_copy.copy()
    # print(HLM_INV_OUT)
    # cv2_imshow(HLM_INV_OUT)
    return HLM_INV_OUT





# ---------------------------------------------------
# def TableMask(r_input): // r_input ; Warped image (inversed) -- mask color to black
#  return src_out // src_out ; 검은색 table mask가 그려진 영상의 copy
# ---------------------------------------------------



def TableMask(r_input):
    src_ = r_input.copy()
    BLACK = (0,0,0)

    h, w = src_.shape[:2]
    dw = 500
    dh = round(dw * 297 / 210)  # A4 용지 크기: 210x297cm

    src_inv = cv2.bitwise_not(src_)
    src_gray = cv2.cvtColor(src_inv, cv2.COLOR_BGR2GRAY)
    _, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(src_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 좌표 변수 등
    # 모서리 점들의 좌표, 드래그 상태 여부
    # 위에서 이미 정의한 함수에서 글로벌 변수로 사용되었던 것들의 존재가 바로 이곳에 있다.

    srcQuad = np.array([[30, 30], [30, h - 30], [w - 30, h - 30], [w - 30, 30]], np.float32) # 내가 선택할 모서리 좌표들 ndarray 반시계방향
    dstQuad = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], np.float32)
    dragSrc = [False, False, False, False] # 현재 어떤 점을 드래그 하고 있나의 Flag

    cont = [c for c in contours if (cv2.contourArea(c) > 500 and cv2.contourArea(c) < (h*w-50000))]

    for pts in cont:
        # 외곽선 근사화
        approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True)
        # 컨벡스가 아니고, 사각형이 아니면 무시
        if not cv2.isContourConvex(approx) or len(approx) != 4:
            continue
        srcQuad = reorderPts(approx.reshape(4, 2).astype(np.float32))

    points = tuple(srcQuad.astype(int))

    cv2.rectangle(src_, points[1], points[3], BLACK, -1)
    src_out = src_.copy()
    # cv2_imshow(src_)
    return src_out


# get_mask


# get_mask -- draw_HardLineMask으로 선을 좀 더 잘 제거한 것에 대한 Masking 작업
# ----------------------------------------------------------------------------
# def get_mask(input_img): // input_img -- draw_HardLineMask()의 output. 딱히 이미지를 내부적으로 복사할 필요 없음. 해당 마스크만 이용할 것
# return image_thresh // 흰색 masking 이미지
# ----------------------------------------------------------------------------




def get_mask(input_img):
    # 이미지를 grayscale로 변환하고 blur를 적용
    # 모서리를 찾기위한 이미지 연산
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 75, 200)

    #plt_imshow(['gray', 'blurred', 'edged'], [gray, blurred, edged])

    # initialize a rectangular kernel that is ~5x wider than it is tall,
    # then smooth the image using a 3x3 Gaussian blur and then apply a blackhat morphological operator to find dark regions on a light background

    kernel = cv2. getStructuringElement ( cv2.MORPH_RECT,  ( 10 , 15 ))
    gray = cv2. GaussianBlur ( edged,  ( 3, 3 ) ,  0 )
    blackhat = cv2. morphologyEx ( gray, cv2.MORPH_BLACKHAT, kernel )


    # compute the Scharr gradient of the blackhat image and scale the result into the range [0, 255]
    grad = cv2.Sobel ( blackhat, ddepth=cv2.CV_32F, dx= 0 , dy= 1 , ksize= -1 )
    grad = np.absolute(grad)
    (minVal, maxVal)  =  (np.min(grad), np.max(grad))
    grad =  (grad - minVal)  /  (  maxVal - minVal  )
    grad =  (grad*255).astype("uint8")


    # apply a closing operation using the rectangular kernel to close gaps in between characters, apply Otsu's thresholding method, and finally a dilation operation to enlarge foreground regions
    grad = cv2.morphologyEx (grad, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.threshold ( grad, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU )[1]
    thresh = cv2.dilate ( thresh, None, iterations= 3 )
    image_thresh = thresh.copy()


    # cv2_imshow (image_thresh)
    return image_thresh



# def get_small_mask(input_img):
# --------------------------------------------------------------------------------
# def get_small_mask(input_img):input_img -- draw_HardLineMask()의 output. 딱히 이미지를 내부적으로 복사할 필요 없음. 해당 마스크만 이용할 것
# return image_thresh // 흰색 masking 이미지 - get_mask보다 작은 mask
# --------------------------------------------------------------------------------


def get_small_mask(input_img):
    # 이미지를 grayscale로 변환하고 blur를 적용
    # 모서리를 찾기위한 이미지 연산
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 75, 200)

    #plt_imshow(['gray', 'blurred', 'edged'], [gray, blurred, edged])

    # initialize a rectangular kernel that is ~5x wider than it is tall,
    # then smooth the image using a 3x3 Gaussian blur and then apply a blackhat morphological operator to find dark regions on a light background

    kernel = cv2. getStructuringElement ( cv2.MORPH_RECT,  ( 5 , 3 ))
    gray = cv2. GaussianBlur ( edged,  ( 3, 3 ) ,  0 )
    blackhat = cv2. morphologyEx ( gray, cv2.MORPH_BLACKHAT, kernel )


    # compute the Scharr gradient of the blackhat image and scale the result into the range [0, 255]
    grad = cv2.Sobel ( blackhat, ddepth=cv2.CV_32F, dx= 0 , dy= 1 , ksize= -1 )
    grad = np.absolute(grad)
    (minVal, maxVal)  =  (np.min(grad), np.max(grad))
    grad =  (grad - minVal)  /  (  maxVal - minVal  )
    grad =  (grad*255).astype("uint8")


    # apply a closing operation using the rectangular kernel to close gaps in between characters, apply Otsu's thresholding method, and finally a dilation operation to enlarge foreground regions
    grad = cv2.morphologyEx (grad, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.threshold ( grad, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU )[1]
    thresh = cv2.dilate ( thresh, None, iterations= 3 )

    image_thresh = thresh.copy()

    # cv2_imshow (image_thresh)
    return image_thresh




# ----------------------------------------------------------------------
# small Mask에 대한 findContour --> 실질적인 text region detecting
# ----------------------------------------------------------------------
# def sMaskContur(small_mask_self): // small_mask_self -- weak line을 masking하기 위해 만들었던 작은 kernel 마스크 이미지
# return // 해당 마스크 별 contour tuple list (조건 하에 findContours 돌리기)
# ----------------------------------------------------------------------

def sMaskContur(small_mask_self, for_drawing_img):
    # gray = cv2.cvtColor(small_mask_self, cv2.COLOR_BGR2GRAY) #흑백화
    # threshing for find contour
    small_mask_self_copy = cv2.bitwise_not(small_mask_self)
    for_drawing_img_copy = for_drawing_img.copy()

    ret, otsu = cv2.threshold(small_mask_self_copy, -1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #오츠 알고리즘 이진화
    # find contours
    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  #윤곽선 검출

    idx = 1; # crop된 이미지가 서로 다른 창에 뜰 수 있도록 idx를 사용
    for cnt in contours:
        if 100000 >cv2.contourArea(cnt) > 30: # 위 사각형 그리기에서 면적 범위를 지정해준 뒤 그 범위 이상일 때만 작동
            x, y, width, height = cv2.boundingRect(cnt) #contours에서 검출된 윤곽선 중 25000의 값을 넘은 선을 감싸는 사각형의 위치와 크기를 얻음
            BBcords.append([x,y,x+width,y+height]) # 좌상단의 x좌표, y좌표, 너비, 높이 순으로 반환된 것을 변환

            cv2.rectangle(for_drawing_img_copy,(x, y),(x + width, y + height), (0, 0, 255), 2) # boundingRect에 저장된 사각형들에 초록색 두께 2인 네모를 그려줌
            idx+=1 # 한 개의 창에 크롭된 이미지를 출력한 후 새로운 창에 출력하기 위해 idx 1증가
            for_drawing_img_copy_return = for_drawing_img_copy.copy()


    # cv2_imshow(std_masked_img_cpy)

    return for_drawing_img_copy_return






# -----------------------------------
# def draw_WeakLineMask (r_input): // r_input ; Warped image (inversed)
#    return cvtBin_Img_out // cvtBin_Img_out ; table line을 굵은 선으로 그린 inversed 이미지 복사본 출력 # 출력된 WeakLinemaskImg는 INV 처리된 copyed img
# -----------------------------------


# ---------------------------------------------------------------------------
# def Masking_src (Masked_INV, image_thresh_self): // Masked_INV ; masking_for_std = get_small_mask(HardLineMaskedIMG)으로 얻은 결과
#             // image_thresh_self = WeakLineMaskImg = draw_WeakLineMask(r_ceta_copy)으로 얻은 흰색 color이미지
# return get_stu_mask // binary로 bitwise_and 시켜 얻은 검은색 bin 이미지
# ---------------------------------------------------------------------------



def Masking_src (Masked_INV, image_thresh_self):
    image_thresh_INV = cv2.cvtColor(image_thresh_self, cv2.COLOR_BGR2GRAY)
    img_INV_copy = image_thresh_INV.copy()
    # img_mask = cv2.bitwise_not(r_cvt)
    get_stu_mask = cv2.bitwise_and(img_INV_copy, Masked_INV)
    # get_stu_mask = cv2.bitwise_or(img_mask, copy_thres) # inv_src와 mask를 or연산하면 테이블만 추출

    # cv2_imshow(get_stu_mask)
    return get_stu_mask








# find range [min,max] and check if item is in the range,
# if item is out of range, it is in next cluster class
# input : 1차원 리스트/배열, output : item마다 class idx가 매겨진 2차원 배열
# answer file을 받아서 warpaffine을 했다고 가정
# -----------------------------------------------------------------
# def ListCluster(self):
#	return tmp
# -----------------------------------------------------------------



# 최종적으로 avg값으로 clustered된 x,y 좌표들의 class 리스트 얻음

# y_coords,x_coords : 리스트([x/y좌표, index],...) / COL, ROW = 사용자지정전역변수
# def getAvgClassXY(y_coords,x_coords,COL,ROW):
# 	return x_class, y_class // 리스트([x/y좌표, index],...) -- 각 좌표들의 작은것부터 순서대로

def getAvgClassXY(y_coords,x_coords,COL,ROW):

    def ListCluster(self):
        min=self[0]
        idx = 0
        tmp = []
        for i in range(1, len(self)-1):
            if (self[i] <= min+Mergdist):
                tmp.append([self[i],idx])
            else:
                min = self[i]
                tmp.append([self[i],idx+1])
                idx+=1
        return tmp

    def GetXcord(self):
        tmp = []
        for i in range(len(self)):
            tmp.append(self[i][0])
        return tmp


    # self = 1차원 리스트 혹은 배열
    def MergedCoords(self,limit):
        count = 0
        tmp = []
        limit = limit
        for i in range(len(self)):
            for j in range(len(self)):
                if abs(self[i] - self[j]) < Mergdist:
                    count += 1
                    if count//4 > limit:
                        tmp.append([self[i],count//4])
                        count = 0
        return tmp



    # get average of each class -- input : list = [[x,class],...], output = [[avg,class]...]
    def avgClass(self):
        avg = 0
        count = 0
        tmp = []
        idxCount = 0

        for i in range(len(self)):
            if self[i][1] == idxCount:
                avg += self[i][0]
                count += 1
            if i==range(len(self)):
                tmp.append([int(avg/count),idxCount])
            else:
                tmp.append([int(avg/count),idxCount])
                idxCount += 1
                avg = self[i][0]
                count = 1

        return tmp



    T_x_list = []
    T_y_list = []

    x_clusted = []
    y_clusted = []

    T_x_list = MergedCoords(x_coords,ROW)
    T_y_list = MergedCoords(y_coords,COL)

    T_x_list.sort(key=lambda x: x[0])
    T_y_list.sort(key=lambda x: x[0])

    T_x_list = GetXcord(T_x_list)
    T_y_list = GetXcord(T_y_list)

    x_clusted = ListCluster(T_x_list)
    y_clusted = ListCluster(T_y_list)

    x_class = []
    y_class = []
    x_class = avgClass(x_clusted)
    y_class = avgClass(y_clusted)

    return x_class, y_class



    # [[파이썬] 배열 - count(),set(),정수를배열로](https://velog.io/@yejinleee/%ED%8C%8C%EC%9D%B4%EC%8D%AC-count)
    # 1차원 배열 정렬 : np.sort(x) // x는 list
    # [R, Python 분석과 프로그래밍의 친구 (by R Friend) :: [Python] numpy array 정렬, 거꾸로 정렬, 다차원 배열 정렬](https://rfriend.tistory.com/357)

    # print(x_class)
    # print("")
    # print(y_class)





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
#