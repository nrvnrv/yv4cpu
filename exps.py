import cv2

mode = 0
cap = 0
link = "rtsp://admin:a123456789@192.168.0.217"
if int(mode) == 0:  # !
    cap = cv2.VideoCapture(link) 
if int(mode) == 1:  # !
    cap = cv2.VideoCapture(link) 
if int(mode) == 2:
    cap = cv2.VideoCapture(0)  

frame_width = int(cap.get(3) / 2) #2
frame_height = int(cap.get(4) / 2) #2

while True:
    ret, frame_read = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,
                               (frame_width, frame_height),
                               interpolation=cv2.INTER_LINEAR)

    cv2.imshow('tca', frame_resized)
    cv2.waitKey(1)  # 3


cap.release()

print(":::Video Write Completed")

