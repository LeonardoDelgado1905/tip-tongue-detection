import cv2

# Load the Haar Cascade Classifier for tongue detection
cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_alt_tree.xml')
#cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_alt.xml')
#cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_alt2.xml')
#cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_default.xml')

# Initialize the KCF tracker
tracker = cv2.TrackerKCF_create()

# Initialize the video capture device
cap = cv2.VideoCapture(0)

# Wait for the camera to warm up
cv2.waitKey(1000)

# Start the video loop
while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect tongues in the grayscale image
    tongues = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If a tongue is detected, start tracking it with KCF
    if len(tongues) > 0:
        # Select the largest tongue as the object of interest
        x, y, w, h = sorted(tongues, key=lambda x: x[2] * x[3])[-1]

        # Initialize the KCF tracker with the current frame and bounding box
        tracker.init(frame, (x, y, w, h))

    # If a tongue is being tracked, update the tracker and draw the bounding box
    if tracker:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with the bounding box
    cv2.imshow('Tongue Detection', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
