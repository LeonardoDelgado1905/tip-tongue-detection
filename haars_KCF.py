import cv2
import os

# Get the absolute path of the current Python script file
script_path = os.path.abspath(__file__)
# Get the directory path of the current Python script file
dir_path = os.path.dirname(script_path)

# Print the current path and directory path
print('Current path:', script_path)
path_to_cascade = f'{dir_path}\haarcascade_frontalface_alt_tree.xml'
print('Directory path:', path_to_cascade)
# Load the Haar Cascade Classifier for tongue detection
cascade = cv2.CascadeClassifier(path_to_cascade)
#cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_alt.xml')
#cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_alt2.xml')
#cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_default.xml')

# Create a KCF tracker object
tracker = cv2.TrackerKCF_create()

# Set up the video capture device (use 0 for default camera)
cap = cv2.VideoCapture(0)

# Initialize the tracking variables
bbox = None
tracking_started = False

while True:
    # Read a new frame from the video capture device
    ret, frame = cap.read()
    
    # If there was an error reading the frame, break out of the loop
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If a face is detected, start tracking the tongue
    if len(faces) > 0:
        # Get the first face detected
        (x, y, w, h) = faces[0]
        
        # If tracking hasn't started yet, initialize the tracker with the face bounding box
        if not tracking_started:
            bbox = (x, y, w, h)
            ok = tracker.init(frame, bbox)
            tracking_started = True
        # If tracking has already started, update the tracker with the current frame and bounding box
        else:
            # Ensure that the tongue is inside the face rectangle before updating the tracker
            if bbox is not None and x <= bbox[0] <= x+w and y <= bbox[1] <= y+h:
                # Convert the bounding box to the format expected by the KCF tracker
                bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
                
                # Update the tracker with the current frame and bounding box
                ok, bbox = tracker.update(frame, bbox)
                
                # If the tracking is successful, draw a rectangle around the tracked object
                if ok:
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 255), 2)
                else:
                    # If the tracking fails, reset the tracking variables
                    bbox = None
                    tracking_started = False
            else:
                # If the tongue is not inside the face rectangle, reset the tracking variables
                bbox = None
                tracking_started = False
    
    # Display the resulting image
    cv2.imshow('Tongue Tracker', frame)
    
    # Wait for a key press and check if the user wants to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
