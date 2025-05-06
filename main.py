import cv2
from ultralytics import YOLO 

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture("videoplayback.mp4")

entry_line_y = 200 
exit_line_y = 300 

person_count = 0
previous_y={}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame,persist=True,classes=0)

    detected_people = {}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0] 
            confidence = box.conf[0]
            person_id=int(box.id[0]) if box.id is not None else None

            if confidence > 0.5 and person_id is not None:
                center_y = (y1 + y2) / 2 
                detected_people[person_id] = center_y

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame,f"person id: {person_id}",(int(x1), int(y1) - 10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)

    for person_id, current_y in detected_people.items():
            if person_id in previous_y:
                previ_p=previous_y[person_id]
                if previ_p<entry_line_y or current_y>=entry_line_y:
                    person_count += 1  

                if previ_p>exit_line_y or current_y<=exit_line_y:
                    person_count -= 1 

    cv2.line(frame, (0, entry_line_y), (frame.shape[1], entry_line_y), (255, 0, 0), 2) 
    cv2.line(frame, (0, exit_line_y), (frame.shape[1], exit_line_y), (0, 0, 255), 2) 
    previous_y=detected_people.copy()

    cv2.putText(frame, f"People Entering: {person_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Live People Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()