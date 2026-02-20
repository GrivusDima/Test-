#Моушн дектексьйон
import cv2

cap = cv2.VideoCapture(0) # Opens the default webcam

ret, frame1 = cap.read() # Grabs the very first frame to use as a baseline
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (5, 5), 5) # Blurring is crucial here to ignore tiny pixel flickers/camera noise
gray1 = cv2.convertScaleAbs(gray1, alpha = 1.2, beta = 50) # Tweaks contrast (alpha) and brightness (beta)

while True:
    ret, frame2 = cap.read() # Grabs the next frame in the video stream
    if not ret:
        print("кадри скінчились")
        break
    
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 5)
    gray2 = cv2.convertScaleAbs(gray2, alpha = 1.2, beta = 50)

    
    diff = cv2.absdiff(gray1, gray2) # Calculates the absolute mathematical difference between the two frames (finds movement)

    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY) # Anything that changed by more than 30 intensity turns bright white, rest goes black

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 800: # Ignores tiny movements (like noise or background flicker)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2) # Draws a box around the moving object

    gray1 = gray2 # Updates the baseline frame for the next loop iteration

    if cv2.waitKey(1) & 0xff == ord("q"): # Quits the loop if 'q' is pressed
        break

    # cv2.imshow("Video", gray2)
    cv2.imshow("Video2", frame2)

cap.release() # Frees up the webcam hardware
cv2.destroyAllWindows()
