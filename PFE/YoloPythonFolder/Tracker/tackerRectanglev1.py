import cv2
import numpy as np

class RectangleTracker:
    def __init__(self):
        self.trackers = []

    def update(self, frame):
        new_trackers = []
        for tracker in self.trackers:
            success, box = tracker.update(frame)
            if success:
                new_trackers.append(tracker)
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)  # Green for active tracker
        self.trackers = new_trackers

    def add_tracker(self, frame, box):
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, box)
        self.trackers.append(tracker)

    def clear_trackers(self):
        self.trackers = []

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def non_max_suppression_lines(lines, overlapThresh):
    if len(lines) == 0:
        return []

    lines = np.array(lines)
    pick = []

    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    lengths = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    idxs = np.argsort(lengths)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / lengths[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return lines[pick].astype("int")

def detect_rectangles(frame, tracker):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Ignore small areas
            continue
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:  # Check if the shape has 4 sides
            x, y, w, h = cv2.boundingRect(approx)
            rectangles.append([x, y, x + w, y + h])

    if len(rectangles) > 0:
        rectangles = np.array(rectangles)
        rectangles = non_max_suppression(rectangles, 0.3)  # Adjust overlap threshold

    tracker.clear_trackers()  # Clear existing trackers
    for (x1, y1, x2, y2) in rectangles:
        tracker.add_tracker(frame, (x1, y1, x2 - x1, y2 - y1))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle

    return frame

def detect_lines_in_roi(frame, roi, canny_threshold1=30, canny_threshold2=130, hough_threshold=50, min_line_length=50, max_line_gap=5):
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is not None:
        lines = lines.reshape(-1, 4)
        lines = non_max_suppression_lines(lines, 0.3)  # Adjust overlap threshold for lines
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1 + x, y1 + y), (x2 + x, y2 + y), (0, 0, 255), 1)  # Draw line in red

    return frame

def main():
    cap = cv2.VideoCapture(0)
    tracker = RectangleTracker()

    # Define the region of interest (ROI)
    roi = (100, 100, 400, 400)  # Example ROI, adjust as needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracker.update(frame)  # Update existing trackers
        frame = detect_rectangles(frame, tracker)  # Detect new rectangles
        frame = detect_lines_in_roi(frame, roi, canny_threshold1=30, canny_threshold2=163, hough_threshold=26, min_line_length=50, max_line_gap=5)

        cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)  # Draw ROI rectangle
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()