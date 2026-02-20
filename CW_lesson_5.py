#Маски, форми дві полоски
import cv2 # Imports OpenCV for image processing
import numpy as np # Imports NumPy for mathematical arrays

# Load the image
img = cv2.imread('images/woman.jpg') # Reads the image file from the folder
img_copy = img.copy() # Makes a backup copy of the image to draw on later

# Convert to HSV (Hue, Saturation, Value)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Changes color format from Blue-Green-Red to Hue-Saturation-Value

# Define the color range we want to isolate
lower = np.array([0, 7, 0]) # The lowest shade of our target color (in HSV)
upper = np.array([179, 255, 255]) # The highest shade of our target color (in HSV)
mask = cv2.inRange(img, lower, upper) # Creates a black-and-white "stencil" where the target color is white

# Apply the mask
img = cv2.bitwise_and(img, img, mask=mask) # Overlays the stencil on the image, keeping only the isolated color visible

# Part 2: Contour Analysis
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finds the outlines of the white shapes in the mask

for cnt in contours: # Starts a loop to look at every single shape found
    area = cv2.contourArea(cnt) # Calculates the surface area of the shape in pixels
    if area > 200:  # If the shape is bigger than 200 pixels (ignores tiny dust/noise)
        perimeter = cv2.arcLength(cnt, True) # Calculates the length of the shape's outline
        M = cv2.moments(cnt) # Calculates a dictionary of mathematical properties for the shape

        if M["m00"] != 0: # Checks to make sure the shape isn't empty (prevents dividing by zero errors)
            cx = int(M["m10"] / M["m00"]) # Calculates the X coordinate of the exact center
            cy = int(M["m01"] / M["m00"]) # Calculates the Y coordinate of the exact center

        x, y, w, h = cv2.boundingRect(cnt) # Gets the coordinates to draw a straight box around the shape
        aspect_ratio = round(w / h, 2) # Divides width by height to see if it's tall, wide, or perfectly square
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2) # Calculates how close the shape is to being a perfect circle

        
        # Determine the shape
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True) # Smooths out the shape's jagged edges to find its main corners (vertices)
        if len(approx) == 3: # If it has exactly 3 corners
            shape = "Trikutnik" # It's a triangle
        elif len(approx) == 4: # If it has exactly 4 corners
            shape = "4jtirikutik" # It's a quadrilateral
        elif len(approx) > 8: # If it has a lot of corners (making it smooth)
            shape = "oval" # It's a circle/oval
        else: # If it's anything else
            shape = "inshe" # It's an "other" shape

        # Draw the results on the screen
        cv2.drawContours(img, [cnt], -1, (255,255,255), 2) # Draws the exact outline in white
        cv2.circle(img_copy, (cx, cy), 4, (255,0,0), -1) # Draws a solid blue dot at the center
        cv2.putText(img_copy, f"{shape}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # Writes the shape name in red text
        cv2.putText(img_copy, f"A:{int(area)} P:{int(perimeter)}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1) # Writes the Area and Perimeter in green
        cv2.putText(img_copy, f"AR:{aspect_ratio} C:{compactness}", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1) # Writes Aspect Ratio and Compactness
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2) # Draws the blue bounding box around the shape

        text_y = y - 5 if y - 5 > 10 else y + 15 # Decides whether to put the final text above or below the box
        text = f"x:{x}, y:{y}, S:{int(area)}" # Prepares text with X, Y coordinates and Area (S)
        cv2.putText(img_copy, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) # Draws the final text on screen

cv2.imshow('Mask', img_copy) # Opens a window to show the final image with all the drawings
cv2.waitKey(0) # Pauses the script until you press a key
cv2.destroyAllWindows() # Closes the window
