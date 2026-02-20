import cv2 
import numpy as np 

image = cv2.imread("images/PGTA_2.jpg") 
print(image.shape) 

# image = cv2.resize(image, (1000, 500)) # Resizes the image to exactly 1000x500 pixels
image = cv2.resize(image, (image.shape[1] * 1, image.shape[0] * 1)) # Dynamically resizes the image (currently stays the same size since it multiplies by 1)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converts the image from standard color to grayscale
print(image.shape) # Prints dimensions again (color channels are gone now)

image = cv2.Canny(image, 100, 100) # Creates contours and turns image colors into binary black/white
print(image.shape) # Prints dimensions of the new binary edge image

kernel = np.ones([1, 7], np.uint8) # Creates a 1x7 pixel matrix (a horizontal line shape) to act as a brush for morphological operations

image = cv2.dilate(image, kernel, iterations=2) # Expands light areas, using the shape defined in the kernel variable
image = cv2.erode(image, kernel, iterations=2) # Shrinks light areas back down (combining this with dilation helps connect broken edges)

cv2.imwrite("images/PGTA_2.jpg", image) 

cv2.imshow('PGTA 1', image) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
