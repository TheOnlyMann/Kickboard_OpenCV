import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt

#define base ArUco marker characteristics
marker_size = 100  # 마커의 한 변 길이 (픽셀 단위)
marker_type = aruco.DICT_6X6_250  # 마커 유형
aruco_dict = aruco.getPredefinedDictionary(marker_type)  # 마커 사전 생성

# 대표적인 HSV 색상 (중심값 기준)
HSV_COLOR_DICT = {
    "RED":     (0,   255, 255),   # Hue 0도 (또는 170도도 빨강 계열)
    #"ORANGE":  (20,  255, 255),   # Hue 20도
    #"YELLOW":  (40,  255, 255),   # Hue 40도
    "GREEN":   (75,  255, 255),   # Hue 75도
    "BLUE":    (110, 255, 255),   # Hue 110도
    #"PURPLE":  (145, 255, 255),   # Hue 145도
}
def get_hsv_color(color_name):
    """
    주어진 색상 이름에 해당하는 HSV 색상을 반환합니다.
    """
    return HSV_COLOR_DICT.get(color_name.upper(), (0, 0, 0))  # 기본값은 검정색
def hsv_to_bgr(hsv_color):
    """
    HSV 색상을 BGR 색상으로 변환합니다.
    """
    bgr_color = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(bgr_color)
def bgr_to_hsv(bgr_color):
    """
    BGR 색상을 HSV 색상으로 변환합니다.
    """
    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    return tuple(hsv_color)
def detect_hsv_color(hsv_tuple):
    """
    주어진 HSV 색상 튜플을 사용하여 가장 가까운 색상 이름을 반환합니다.
    """
    min_diff = float('inf')
    closest_color = None
    for color_name, hsv_color in HSV_COLOR_DICT.items():
        diff = np.linalg.norm(np.array(hsv_tuple) - np.array(hsv_color))
        if diff < min_diff:
            min_diff = diff
            closest_color = color_name
    return closest_color

def create_aruco_marker(marker_id, verbose=True):
    """
    주어진 ID와 길이를 가진 ArUco 마커를 생성합니다.
    """
    # ArUco 마커 생성
    marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)  # 길이를 픽셀로 변환
    # 이미지 저장
    cv2.imwrite(f"aruco_marker_{marker_id}.png", marker_img)
    print(f"✅ ArUco 마커 ID {marker_id} 저장됨: aruco_marker_{marker_id}.png")
    if verbose:
        plt.imshow(marker_img, cmap='gray')
        plt.axis('off')
        plt.title(f"ArUco Marker {marker_id}")
        plt.show()
    return marker_img

def bg_img_setup(colors, bg_img):
    raise NotImplementedError("bg_img_setup 함수는 구현되지 않았습니다.")

def create_colored_aruco_marker(marker_id, color_names, verbose=True):
    """
    마커 ID와 두 개의 색상 이름을 받아서, 마커를 양쪽 색상 배경 위에 출력합니다.
    """
    assert len(color_names) == 2, "color_names는 정확히 2개의 색상명을 가져야 합니다."

    # 마커 생성
    marker_size = 200
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # HSV → BGR 색 변환
    bgr_colors = [hsv_to_bgr(get_hsv_color(name)) for name in color_names]

    # 배경 이미지 설정
    border = 50
    total_size = marker_size + 2 * border
    bg_img = np.zeros((total_size, total_size, 3), dtype=np.uint8)

    # 좌/우로 배경 색 나누기
    mid_x = total_size // 2
    bg_img[:, :mid_x] = bgr_colors[0]  # 왼쪽
    bg_img[:, mid_x:] = bgr_colors[1]  # 오른쪽

    # 중앙에 마커 삽입
    bg_img[border:border + marker_size, border:border + marker_size] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    # 텍스트로 색상 이름 추가
    for i, name in enumerate(color_names):
        pos_x = 10 if i == 0 else mid_x + 10
        cv2.putText(bg_img, name, (pos_x, total_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 저장 및 출력
    filename = f"colored_aruco_marker_{marker_id}_{color_names[0]}_{color_names[1]}.png"
    cv2.imwrite(filename, bg_img)
    print(f"✅ 저장됨: {filename}")

    # 표시
    if verbose:
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Colored ArUco Marker {marker_id} ({color_names[0]}, {color_names[1]})")
        plt.show()
    return bg_img


def create_color_aruco_marker(marker_id, fg_color=(0, 0, 0), bg_color=(255, 255, 255), marker_size=200, verbose=True):
    """
    색상 커스터마이징된 ArUco 마커 생성 (인식은 grayscale 기준으로 가능)
    fg_color: 마커 내부 (원래는 검정)
    bg_color: 마커 배경 (원래는 흰색)
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    marker_bw = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # 컬러 마커 생성
    marker_colored = np.zeros((marker_size, marker_size, 3), dtype=np.uint8)
    marker_colored[:, :] = bg_color
    marker_colored[marker_bw == 0] = fg_color

    filename = f"color_aruco_marker_{marker_id}_{fg_color}_{bg_color}.png"
    cv2.imwrite(filename, marker_colored)
    print(f"✅ 저장됨: {filename}")
    if verbose:
        plt.imshow(marker_colored)
        plt.axis('off')
        plt.title(f"Color ArUco Marker {marker_id} ({fg_color}, {bg_color})")
        plt.show()

    return marker_colored
