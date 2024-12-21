import numpy as np
import cv2
from typing import Any

Image = cv2.mat_wrapper.Mat | np.ndarray[Any, np.dtype]


def get_mhi(prev_img: Image, next_img: Image) -> Image:
    """
    두 이미지로부터 모션 히스토리 이미지를 계산해 반환하는 함수
    :param prev_img: 이전 이미지
    :param next_img: 이후 이미지
    :return: 모션 히스토리 이미지
    """
    diff = cv2.absdiff(prev_img, next_img)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return gray_diff


def get_mhis(video: cv2.VideoCapture) -> list[Image]:
    """
    비디오 데이터를 모션 히스토리 이미지의 리스트로 반환하는 함수
    :param video: cv2 비디오
    :return: 모션 히스토리 이미지 리스트
    """
    result = []

    prev_img = None
    while True:
        ret, curr_img = video.read()

        if not ret:
            break

        # 모션 히스토리 이미지 생성
        if prev_img is not None:
            mhi = get_mhi(prev_img, curr_img)
            result.append(mhi)

        prev_img = curr_img

    return result


if __name__ == "__main__":
    def main():
        path = "dataset/walking/person01_walking_d1_uncomp.avi"
        video = cv2.VideoCapture(path)
        mhis = get_mhis(video)
        for mhi in mhis:
            cv2.imshow("MHI", mhi)
            cv2.waitKey(0)

    main()
