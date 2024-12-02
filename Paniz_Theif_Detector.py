#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

# Start video capture object
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

# start capturing after 3 seconds
cv2.waitKey(3000)

# Capture the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Failed to capture image.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Convert the frame to grayscale
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Save the grayscale image as a reference background
cv2.imwrite('background.png', gray_frame)

# Load the reference background image
reference_background = cv2.imread('background.png', cv2.IMREAD_GRAYSCALE)

# Save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('motion_detection_output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    # Capture the current frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between the reference background and the current frame
    diff = cv2.absdiff(reference_background, gray_frame)
    
    # Thresholding 
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Find contours 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if motion is detected
    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > 500: 
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
            motion_detected = True

    if motion_detected:
        cv2.putText(frame, "UNSAFE", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Motion Detection', frame)
    
    # Write the frame to the output video
    out.write(frame)

    # Exit if 'x' is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the capture and writer objects
cap.release()
out.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()

