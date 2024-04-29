from ultralytics import YOLO
import cv2
import math
from pygame import mixer

cap = cv2.VideoCapture('0914.mp4')
model = YOLO('best.pt')

classnames = ['New folder/fire']

mixer.init()
mixer.music.load('alarm.mp3')
mixer.music.set_volume(0.9)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (740, 480))

        result = model(frame, stream=True)
        confi = 0

        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                confi = confidence
                Class = int(box.cls[0])
                if confidence > 20:
                    mixer.music.play()
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cv2.putText(frame, f'Fire {confidence}%', (x1 + 8, y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            if confi < 20:
                mixer.music.stop()

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    else:
        print("Invalid frame")
        break

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()