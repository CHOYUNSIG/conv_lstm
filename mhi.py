import cv2
import numpy as np


def get_mhi(prev_img, curr_img):
    """
    이미지 데이터를 모션 히스토리 이미지로 변환해 반환하는 함수
    """
    # 이전 이미지와 현재 이미지의 차이를 계산
    diff = cv2.absdiff(prev_img, curr_img)

    # 차이 이미지를 그레이스케일로 변환
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 이진화하여 모션을 강조
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    # 모션 히스토리 이미지 생성
    mhi = np.zeros_like(prev_img, dtype=np.float32)
    mhi[thresh > 0] = 1.0  # 모션이 감지된 부분에 1.0 할당

    return mhi


def get_mhis(video):
    pass
