import cv2
import numpy as np

def detect_lines(frame, canny_threshold1, canny_threshold2, hough_threshold, min_line_length, max_line_gap, roi):
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Détecter les bords avec Canny