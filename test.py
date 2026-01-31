import cv2
cap = cv2.VideoCapture(0)
print("Opened:", cap.isOpened())
cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
cv2.resizeWindow("TEST", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("TEST", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
