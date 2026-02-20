#Рісавашкі!1!1!:PP
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Red color wraps around the HSV spectrum (ends at 180, starts at 0), so we need two separate masks to catch all reds
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

points = [] # A list to store the center coordinates of our tracked object over time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Flips the frame horizontally so it acts like a mirror
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1) # Finds the lower red hues
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2) # Finds the upper red hues

    mask = cv2.bitwise_or(mask1, mask2) # Combines both red masks together

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000: # Only tracks reasonably large red objects
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                points.append((cx, cy)) # Saves the center point to our tracking list
            
            # Loops through all the saved points to draw a trailing line
            for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                    continue
                cv2.line(frame, points[i - 1], points[i], (255,0,0), 2) # Connects the history dots
                
    cv2.imshow("video", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
