#Маски, форми дві полоски
import cv2
import numpy as np

img = cv2.imread('images/woman.jpg')
img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Converts to HSV (Hue, Saturation, Value) which is better for color tracking


# Defining the lower and upper bounds for the specific color we want to isolate
lower = np.array([0, 7, 0])
upper = np.array([179, 255, 255])
mask = cv2.inRange(img, lower, upper) # Creates a black/white mask where the targeted color is white

img = cv2.bitwise_and(img, img, mask=mask) # Applies the mask to show only the isolated object in its original colors

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        perimeter = cv2.arcLength(cnt, True) # Calculates the length of the contour boundary (True = closed shape)
        M = cv2.moments(cnt) # Calculates mathematical characteristics of the shape/size

        if M["m00"] != 0: # Prevents division by zero
            # Calculates the (x, y) coordinates of the object's center of mass (centroid)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2) # Width divided by height to help figure out shape proportions
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2) # Measures roundness (closer to 1.0 means more circular)

        
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True) # Smoothes the contour down into basic geometric vertices
        
        # Guesses the shape based on the number of corners (vertices) left after smoothing
        if len(approx) == 3:
            shape = "Trikutnik"
        elif len(approx) == 4:
            shape = "4jtirikutik"
        elif len(approx) > 8:
            shape = "oval"
        else:
            shape = "inshe"

        cv2.drawContours(img, [cnt], -1, (255,255,255), 2)
        cv2.circle(img_copy, (cx, cy), 4, (255,0,0), -1) # Draws the center point
        cv2.putText(img_copy, f"{shape}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img_copy, f"A:{int(area)} P:{int(perimeter)}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(img_copy, f"AR:{aspect_ratio} C:{compactness}", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

        text_y = y - 5 if y - 5 > 10 else y + 15
        text = f"x:{x}, y:{y}, S:{int(area)}"
        cv2.putText(img_copy, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('Mask', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
