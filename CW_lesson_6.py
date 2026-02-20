#Моушн дектексьйон
import cv2 # imports cv2

cap = cv2.VideoCapture(0) # turn on webka

# Set up the very first frame
ret, frame1 = cap.read() # Takes a single picture from the webcam to use as our "background"
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) # Turns that first picture grayscale
gray1 = cv2.GaussianBlur(gray1, (5, 5), 5) # Blurs it to ignore tiny camera static
gray1 = cv2.convertScaleAbs(gray1, alpha = 1.2, beta = 50) # Brightens the image to help shadows stand out

while True: # Starts an endless loop to process the live video
    ret, frame2 = cap.read() # Takes the *next* picture from the webcam
    if not ret: # If the webcam stops working or the video ends
        print("кадри скінчились") # Prints "frames finished"
        break # Breaks out of the endless loop
    
    # Process the new frame identically to the first one
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) # Turns the new picture grayscale
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 5) # Blurs it
    gray2 = cv2.convertScaleAbs(gray2, alpha = 1.2, beta = 50) # Brightens it

    
    # Find the movement
    diff = cv2.absdiff(gray1, gray2) # Compares Frame 1 and Frame 2. If a pixel changed, it gets marked.
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY) # If a pixel changed enough (value > 30), it turns pure white. Otherwise, pure black.

    # Draw boxes around the movement
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finds the outlines of the white (moving) shapes
    for cnt in contours: # Loops through every moving shape
        if cv2.contourArea(cnt) > 800: # If the moving shape is big enough (ignores dust/flickering lights)
            x, y, w, h = cv2.boundingRect(cnt) # Gets the coordinates for a box
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2) # Draws a green box around whatever moved

    gray1 = gray2 # IMPORTANT: Makes the current frame the new "background" for the next loop to compare against

    if cv2.waitKey(1) & 0xff == ord("q"): # Checks 1 millisecond per loop to see if you pressed the 'q' key
        break # If you pressed 'q', it breaks the loop to close the program

    # cv2.imshow("Video", gray2) # (Commented out) Would show the processed grayscale video
    cv2.imshow("Video2", frame2) # Shows the live color video with the green tracking boxes drawn on it

cap.release() # Turns off the webcam so other apps can use it
cv2.destroyAllWindows() # Closes the video windows
