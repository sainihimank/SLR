import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands module and utilities for drawing landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up MediaPipe Hands with a static image mode and a minimum detection confidence of 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory where the dataset is stored
DATA_DIR = './data'

data = []  # To store the processed landmark data
labels = []  # To store the corresponding labels for each data point

# Iterate through each directory (representing each class) in the dataset
for dir_ in os.listdir(DATA_DIR):
    # Iterate through each image in the current class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to store landmarks for the current image
        x_ = []  # List to store x-coordinates of landmarks
        y_ = []  # List to store y-coordinates of landmarks

        # Load the image and convert it from BGR to RGB (required for MediaPipe)
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x and y coordinates of each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the coordinates and append to the auxiliary data list
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalize x-coordinates
                    data_aux.append(y - min(y_))  # Normalize y-coordinates

            data.append(data_aux)  # Append the processed landmarks to the data list
            labels.append(dir_)  # Append the corresponding label (directory name)

# Save the processed data and labels into a pickle file
with open('data_new.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
