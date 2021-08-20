import cv2

from WebcamVideoStream import WebcamVideoStream

vs = WebcamVideoStream('rtsp://admin:a123456789@192.168.0.217').start()
while True :
    frame = vs.read()
    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) == 27 :
        break

vs.stop()
cv2.destroyAllWindows()