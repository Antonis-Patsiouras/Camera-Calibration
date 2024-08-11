import cv2 ,subprocess 

# Open a connection to the web camera (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open web camera.")
    exit()

# Read and display video frames
    # Capture frame-by-frame
while True:
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('r'):
        # Λήψη εικόνας 1
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cv2.imwrite(r"C:\Users\apats\OneDrive\Documents\MSc Robotics\P201_1\Project2\images\image1.jpg", frame)
        print("Image 1 captured")
    if cv2.waitKey(1) & 0xFF == ord('t'):
        # Λήψη εικόνας 2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cv2.imwrite(r"C:\Users\apats\OneDrive\Documents\MSc Robotics\P201_1\Project2\images\image2.jpg", frame)
        print("Image 2 captured")
        break
    if not ret:
        print("Error: Failed to capture image")
        break

    # Display the resulting frame
    cv2.imshow('Web Camera', frame)

    # Press 'q' on the keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
subprocess.run(["python", "calibration.py"])