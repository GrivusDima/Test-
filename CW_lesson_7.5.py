import cv2
import numpy as np

# Load the image of the candies (replace 'images/candies.jpg' with your actual file path)
img = cv2.imread('images/candies.jpg')

# Convert the image from standard BGR (Blue, Green, Red) to HSV (Hue, Saturation, Value).
# HSV is much better for color detection because 'Hue' separates the pure color from lighting and shadows (Value).

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# --- DEFINE COLOR RANGES ---
# We set a 'lower' and 'upper' limit for each color to catch different shades and lighting variations.

# Candy 1: Red
# Red is unique in OpenCV's HSV because it wraps around the 0-180 degree circle. 
# It exists at the very beginning (0-10) and the very end (160-180), so we need two separate ranges.
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Candy 2: Green (Hue ranges roughly from 40 to 90)
lower_green = np.array([40, 50, 50])
upper_green = np.array([90, 255, 255])

# Candy 3: Blue (Hue ranges roughly from 100 to 140)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Candy 4: Yellow (Hue ranges roughly from 20 to 40)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# --- CREATE MASKS (STENCILS) ---
# cv2.inRange checks every single pixel. If the pixel's color falls within our lower/upper limits, 
# it turns it pure white. If not, it turns it pure black.

# Process Red: Create masks for both ends of the red spectrum, then combine them.
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2) # bitwise_or means: if a pixel is white in mask 1 OR mask 2, keep it white.

# Process Green, Blue, and Yellow
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Combine all the masks into one
mask_combined = (mask_red | mask_green | mask_blue | mask_yellow)

# --- BUNDLE DATA FOR PROCESSING ---
# We create a list of groups to keep our code clean and avoid repeating the drawing steps four times. 
# Each group contains: (The specific color mask, "The Text Label", (B, G, R box color))
candy_types = [
    (mask_red, "Red", (0, 0, 255)),        # Red box in BGR format
    (mask_green, "Green", (0, 255, 0)),    # Green box in BGR format
    (mask_blue, "Blue", (255, 0, 0)),      # Blue box in BGR format
    (mask_yellow, "Yellow", (0, 255, 255)) # Yellow box in BGR format
]

# --- FIND CANDIES AND DRAW BOXES ---

# We loop through our bundled list one color at a time.
for mask, color_name, box_color in candy_types:
    
    # Find the outlines (contours) of the white shapes in the current color's mask.
    # RETR_EXTERNAL ensures we only get the outside edge of the candy, ignoring any internal reflections or holes.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    for cnt in contours:
        area = cv2.contourArea(cnt) # Measure how many pixels are inside the shape.
        
        if area > 500: # Filter out tiny specks of background noise (you may need to tweak this number based on your image size).
            
            
            # Calculate the X, Y coordinates, Width, and Height needed to draw a straight box around the contour.
            x, y, w, h = cv2.boundingRect(cnt) 
            
            # Draw the rectangle on the ORIGINAL image.
            # Syntax: image, top-left corner, bottom-right corner, color, line thickness
            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
            
            # Add the text label just above the box (y - 10 pushes it up slightly so it doesn't overlap the line).
            cv2.putText(img, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

# Show the final result
cv2.imshow("Detected Candies", img) 
cv2.waitKey(0) # Keep the window open until you press a key
cv2.destroyAllWindows() # Clean up and close windows
