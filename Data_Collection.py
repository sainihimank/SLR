import os
import cv2

# Directory where the data will be saved
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Create the data directory if it doesn't exist

number_of_classes = 26  # Number of different classes (e.g., A-Z)
dataset_size = 100  # Number of images per class

# Capture video from the default camera (index 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    # Loop through each class
    for j in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)  # Create a directory for each class if it doesn't exist

        print('Collecting data for class {}'.format(j))

        # Prompt the user to start collecting data
        while True:
            ret, frame = cap.read()  # Capture a frame from the camera
            if not ret:
                print("Error: Could not read frame from camera.")
                break

            # Display a message on the frame
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)  # Show the frame in a window

            # Wait for the user to press 'q' to start capturing images
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()  # Capture a frame from the camera
            if not ret:
                print("Error: Could not read frame from camera.")
                break

            cv2.imshow('frame', frame)  # Show the frame in a window
            cv2.waitKey(25)

            # Save the captured frame as an image file
            image_path = os.path.join(class_dir, 'b_{}.jpg'.format(counter))
            cv2.imwrite(image_path, frame)

            counter += 1  # Increment the counter

    # Release the camera and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
