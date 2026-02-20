#Рісавашкі!1!1!:PP
import cv2 
import numpy as np 

cap = cv2.VideoCapture(0) 

# Red is tricky in HSV because it exists at the very start (0) and very end (180) of the spectrum
# So, we define the "low" reds here...
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
# ...and the "high" reds here.
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

points = [] # Creates an empty list. We will store the X,Y coordinates of the object here every frame.

while True: # Starts the live video loop
    ret, frame = cap.read() # Grabs a frame from the webcam
    if not ret: # If the camera fails
        break # Stop the loop

    frame = cv2.flip(frame, 1) # Flips the video horizontally so moving left feels like moving left (mirror mode)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Converts the frame to HSV colors
    
    # Create the red masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1) # Finds the low reds
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2) # Finds the high reds
    mask = cv2.bitwise_or(mask1, mask2) # Glues both red masks together into one master red mask

    
    # Find the red object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finds the outline of the red objects
    for cnt in contours: # Loops through any red items found
        area = cv2.contourArea(cnt) # Checks the size of the red item
        if area > 2000: # Only activates if the red item is large (like a red cup or phone case)
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2) # Draws a green outline exactly around the red object
            M = cv2.moments(cnt) # Calculates the math properties of the shape
            if M['m00'] != 0: # Prevents zero-division errors
                cx = int(M["m10"] / M["m00"]) # Finds the exact X center of the red object
                cy = int(M["m01"] / M["m00"]) # Finds the exact Y center of the red object

                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1) # Draws a blue dot precisely in the center of the object
                points.append((cx, cy)) # Saves that center point's coordinates into our 'points' list
            
            # Draw the trail
            for i in range(1, len(points)): # Loops through all the saved points in history
                if points[i - 1] is None or points[i] is None: # Skips empty points if any exist
                    continue
                cv2.line(frame, points[i - 1], points[i], (255,0,0), 2) # Plays "connect the dots" by drawing a line from the previous point to the current point
                
    cv2.imshow("video", frame) # Shows the normal video with the tracking trail drawn on it
    cv2.imshow("mask", mask) # Shows the black and white stencil of the red items

    if cv2.waitKey(1) & 0xFF == ord("q"): # Waits for you to press 'q'
        break # Stops the program

cap.release() # Releases the webcam
cv2.destroyAllWindows() # Closes the windows
