import cv2
import numpy as np

img = cv2.imread('images/cukerki.jpg')
scale = 4
img = cv2.resize(img, (int(img.shape[1] // scale), int(img.shape[0] // scale)))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_orange = np.array([0, 100, 100])
upper_orange = np.array([25, 255, 255])

lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])


mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

mask_all = (mask_orange | mask_blue | mask_yellow)

candy_types = [
    (mask_orange, "Pomarancheva", (0, 0, 255)),
    (mask_blue, "Synia", (255, 0, 0)),
    (mask_yellow, "Zhofta", (0, 255, 255))]

for mask, color_name, box_color in candy_types:

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(img, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

cv2.imwrite("images/cukerki_bacheni.jpg", img)
cv2.imshow("Bacheni Cukerki (Black and White version)", mask_all)
cv2.imshow("Bacheni Cukerki", img)

cv2.waitKey(0)
cv2.destroyAllWindows()