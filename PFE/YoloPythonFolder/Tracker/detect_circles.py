import cv2
print(cv2.__version__)
import numpy as np

def detect_circles(frame, dp, min_dist, param1, param2, min_radius, max_radius):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Détecter les cercles avec la transformée de Hough
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    
    # Dessiner les cercles détectés
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(frame, center, radius, (255, 0, 0), 2)  # Dessiner en bleu

    return frame
