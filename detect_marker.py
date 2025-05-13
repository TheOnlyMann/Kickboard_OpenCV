import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from generate_marker import detect_hsv_color, HSV_COLOR_DICT
marker_type = aruco.DICT_6X6_250  # ArUco marker type

def get_surrounding_hsv_color(image, pts, size=30):
    """
    Extracts two dominant HSV colors from the area surrounding an ArUco marker.
    Args:
        image: Original BGR image
        pts: 4 corner points of the detected ArUco marker
        size: Padding area around the marker to include in the surrounding region
    Returns:
        Tuple of 2 dominant HSV color tuples
    """
    x, y, w, h = cv2.boundingRect(pts.astype(int))
    x1, y1 = max(x - size, 0), max(y - size, 0)
    x2, y2 = min(x + w + size, image.shape[1]), min(y + h + size, image.shape[0])

    # Extract ROI with surroundings
    surrounding_area = image[y1:y2, x1:x2]
    hsv_area = cv2.cvtColor(surrounding_area, cv2.COLOR_BGR2HSV)

    # Mask the marker area to exclude it
    mask = np.ones(hsv_area.shape[:2], dtype=np.uint8) * 255
    shifted_pts = pts.astype(int) - np.array([x1, y1])
    cv2.fillPoly(mask, [shifted_pts], 0)  # black out the marker

    # Apply mask and reshape
    masked_hsv = hsv_area[mask == 255]
    if masked_hsv.size == 0:
        return ((0, 0, 0), (0, 0, 0))

    # KMeans to find two dominant HSV clusters
    kmeans = KMeans(n_clusters=2, n_init=10)
    try:
        kmeans.fit(masked_hsv)
        colors = kmeans.cluster_centers_
        return tuple(map(tuple, colors.astype(int)))
    except Exception as e:
        print("KMeans failed:", e)
        mean_color = masked_hsv.mean(axis=0).astype(int)
        return (tuple(mean_color), tuple(mean_color))
    
# Detect markers and nearby color
def detect_aruco_with_color(image_path, verbose=True):
    image = cv2.imread(image_path)
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
            hsv = get_surrounding_hsv_color(image, pts)
            print(f"Surrounding HSV for ID {id_val}: {hsv}")
            color_name = [detect_hsv_color(hsv[0]), detect_hsv_color(hsv[1])]
            
            results.append((id_val, color_name))

            # Draw marker + color
            cv2.polylines(image, [pts.astype(int)], True, (0, 255, 0), 2)
            x, y = int(pts[0][0]), int(pts[0][1])
            label = f"ID:{id_val} {color_name}"
            cv2.putText(image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if verbose:
            cv2.imshow("Detected ArUco + Color", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"Detected {len(ids)} markers.")
        for id_val, color_name in results:
            print(f"Marker ID: {id_val}, Colors: {color_name}")

    else:
        print("No markers detected.")
    
    return results

  
# Detect markers and nearby color
def detect_aruco(image_path, verbose=True):
    image = cv2.imread(image_path)
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

# Example usage
detect_aruco_with_color("20250512_145530.jpg", verbose=True)