# Обробка фото та виявлення контурів
import cv2
import numpy as np

img = cv2.imread('images/PGTA_1.jpg')

scale = 1
img = cv2.resize(img, (int(img.shape[1] // scale), int(img.shape[0] // scale))) # Scales image down by dividing dimensions by 'scale'

img_copy = img.copy() # Keeps the original safe while we process the copy

img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (9, 9), 0) # Blurs the image slightly to reduce background noise before edge detection

img_copy = cv2.equalizeHist(img_copy) # Boosts contrast so edges stand out more clearly
img_edges = cv2.Canny(img_copy, 50, 50)


# RETR_EXTERNAL gets only the outer boundaries, CHAIN_APPROX_SIMPLE compresses the contour points to save memory
contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150: # Filters out tiny noise specs
        x, y, w, h = cv2.boundingRect(cnt) # Gets the coordinates and size for a straight rectangle around the contour

        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2) # Draws the actual exact contour shape

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # Draws the bounding box

        text_y = y - 10 if y-10 > 20 else y + 10 # Puts text above the box, or inside if it's too close to the top edge
        text = f'x:{x} y:{y} w:{w} h:{h}'
        cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('edges', img_edges) # Shows the black/white edge map

cv2.imshow('img', img) # Shows the original image with drawings on top
cv2.waitKey(0)
cv2.destroyAllWindows()
