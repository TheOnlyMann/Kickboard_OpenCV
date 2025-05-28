import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from generate_marker import detect_hsv_color, HSV_COLOR_DICT
marker_type = aruco.DICT_6X6_250  # ArUco marker type

# Detect markers and nearby color
def detect_aruco(image, verbose=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.getPredefinedDictionary(marker_type)
    parameters = aruco.DetectorParameters()

    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.adaptiveThreshConstant = 7

    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.03

    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX


    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    results = []

    if ids is not None:
        for i, corner in enumerate(corners):
            id_val = ids[i][0]
            pts = corner[0]

            # Get surrounding HSV
            
            results.append(id_val)

            # Draw marker + color
            cv2.polylines(image, [pts.astype(int)], True, (0, 255, 0), 2)
            x, y = int(pts[0][0]), int(pts[0][1])
            label = f"ID:{id_val}"
            cv2.putText(image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if verbose:
            cv2.imshow("Detected ArUco + Color", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"Detected {len(ids)} markers.")
        for id_val in results:
            print(f"Marker ID: {id_val}")

    else:
        print("No markers detected.")
    
    return results

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


def sliding_crop_zoom_detection_colorless(image, crop_size=500, step=250, zoom=2, verbose=False):
    h, w = image.shape[:2]
    detected = []

    for y in range(0, h - crop_size + 1, step):
        for x in range(0, w - crop_size + 1, step):
            crop = image[y:y + crop_size, x:x + crop_size]
            resized = cv2.resize(crop, (crop_size * zoom, crop_size * zoom))
            results = detect_aruco(resized, verbose=verbose)

            if results:
                for id_val in results:
                    detected.append({
                        "id": id_val,
                        "roi": (x, y, crop_size, crop_size)
                    })

    return detected

# Example usage

input_image = cv2.imread("20250527_161854.jpg")
detected = sliding_crop_zoom_detection_colorless(preprocess_image(input_image), crop_size=200, step=100, zoom=5, verbose=False)
for item in detected:
    print(f"Detected Marker ID: {item['id']} at ROI: {item['roi']}")

