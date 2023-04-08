import cv2
import math
from class_names import classes
from sort import *       # https://github.com/abewley/sort/blob/master/sort.py
from ultralytics import YOLO


path = input("Enter your Relative path of file")

# loading the model with pretrained weights
model = YOLO('yolov8m.pt')


# counters for tracking number of vehicles going Up and Down
ids_up = []
ids_down = []


# Creating the SORT tracker object
tracker = Sort(max_age=40, min_hits=4, iou_threshold=0.3)

# Storing the end points of reference line in pts as a list of tuple of (x, y)
pts = []
c = 0



cap = cv2.VideoCapture(path)


flag, img = cap.read()

cap.release()

cv2.putText(img, 'Select the end points of your reference line', (290, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
def click_event(event, x, y, flags, params):
    global c

    if event == cv2.EVENT_LBUTTONDOWN:
        c = c + 1
        if c<3:
            cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
            pts.append((x, y))

        if c == 2:
            cv2.line(img, pts[0], pts[1], (0, 0, 255), 2)


# create a window
cv2.namedWindow('Point Coordinates')

# bind the callback function to window
cv2.setMouseCallback('Point Coordinates', click_event)

# display the image
while True:
    cv2.imshow('Point Coordinates', img)
    # k = cv2.waitKey(1) & 0xFF
    # if k == 27:
    #     break
    cv2.waitKey(1)
    if c>2:
        break
cv2.destroyAllWindows()


out = cv2.VideoWriter('Output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (img.shape[1], img.shape[0]))

cap = cv2.VideoCapture(path)


while cap.isOpened():
    flag, frame = cap.read()

    img = frame

    cv2.line(img, (pts[0]), pts[1], (0, 0, 255), 2)


    results = model(frame, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1),int(x2),int(y2)
            conf = box.conf[0]
            conf = (math.ceil(conf*100))/100


            cls_index = int(box.cls[0])
            class_name = classes[cls_index]

            if class_name == 'car' or class_name == 'bicycle' or class_name == 'motorcycle' or class_name == 'bus' or class_name == 'truck' and conf > 0.50:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)



    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))

        cv2.putText(img, f'{Id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        w, h = x2 - x1, y2 - y1

        cx, cy = x1+ w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), -1)

        if pts[0][0] < cx < pts[1][0] and pts[0][1] -25 < cy < pts[0][1] + 25:
            if  math.fabs(cx - pts[0][0]) > math.fabs(pts[1][0] - cx):
                if ids_up.count(Id) == 0:
                    ids_up.append(Id)
            else:
                if ids_down.count(Id) == 0:
                    ids_down.append(Id)

    cv2.rectangle(img, (165, 20), (342, 120), (255, 0, 255), -1)
    cv2.putText(img, f'Up Count: {len(ids_up)}', (170, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 12), 2)
    cv2.putText(img, f'Down Count: {len(ids_down)}', (170, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 12), 2)


    cv2.imshow('video', img)
    out.write(img)
    cv2.waitKey(1)




out.release()
cap.release()