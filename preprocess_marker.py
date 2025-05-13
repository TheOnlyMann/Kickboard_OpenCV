import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt

from detect_marker import detect_aruco_with_color, detect_aruco

def preprocess_image(image, target_size=(1000, 1000), gamma=1.4):
    """
    ArUco 인식 정확도 향상을 위한 이미지 보정 함수.
    - 리사이즈
    - 밝기 대비 조정 (Y 채널 평활화)
    - 감마 보정
    """
    # 리사이즈 (이미지가 너무 크면 감지 속도↓)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # YCrCb 평활화
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    image_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    # 감마 보정
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    image_gamma = cv2.LUT(image_eq, table)
    

    return image_gamma




def sliding_crop_zoom_detection(image, crop_size=500, step=250, zoom=2, verbose=False):
    h, w = image.shape[:2]
    detected = []

    for y in range(0, h - crop_size + 1, step):
        for x in range(0, w - crop_size + 1, step):
            crop = image[y:y + crop_size, x:x + crop_size]
            resized = cv2.resize(crop, (crop_size * zoom, crop_size * zoom))
            cv2.imwrite(f"zoomed_crop.jpg", resized)

            results = detect_aruco_with_color("zoomed_crop.jpg", verbose=verbose)

            if results:
                for id_val, color_name in results:
                    detected.append({
                        "id": id_val,
                        "corner": color_name,
                        "roi": (x, y, crop_size, crop_size)
                    })

    return detected

def sliding_crop_zoom_detection_colorless(image, crop_size=500, step=250, zoom=2, verbose=False):
    h, w = image.shape[:2]
    detected = []

    for y in range(0, h - crop_size + 1, step):
        for x in range(0, w - crop_size + 1, step):
            crop = image[y:y + crop_size, x:x + crop_size]
            resized = cv2.resize(crop, (crop_size * zoom, crop_size * zoom))
            cv2.imwrite(f"zoomed_crop.jpg", resized)

            results = detect_aruco("zoomed_crop.jpg", verbose=verbose)

            if results:
                for id_val in results:
                    detected.append({
                        "id": id_val,
                        "roi": (x, y, crop_size, crop_size)
                    })

    return detected