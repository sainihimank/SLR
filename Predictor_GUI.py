import cv2
import mediapipe as mp
import numpy as np
import pickle
import tkinter as tk
from tkinter import *
import customtkinter
from PIL import Image, ImageTk
from collections import Counter
import pyttsx3

# Load the pre-trained model from the pickle file
model_dict = pickle.load(open('./model_new.p', 'rb'))
model = model_dict['model']

# Initialize webcam for capturing video
cap = cv2.VideoCapture(0)

# Initialize MediaPipe for hand detection and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up the hand detection model with desired parameters
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary mapping labels to alphabets (A-Z)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Initialize variables for handling frame capture and text processing
kap, vcap = cap.read()
frame_tk = vcap
word = ""
sentence = ""
blank_frame_start_time = None
paused = False
predicted_buffer = []
incrementor = 0

def update_ui():
    """Function to update the UI with the webcam feed and predictions."""
    global frame_tk, word, paused, predicted_buffer
    try:
        ret, frame = cap.read()  # Capture frame from webcam
        H, W, _ = frame.shape  # Get frame dimensions

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB

        results = hands.process(frame_rgb)  # Process the frame with MediaPipe Hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract hand landmarks and normalize coordinates
                data_aux = []
                x_ = []
                y_ = []

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Define bounding box for the detected hand
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Predict the alphabet based on the hand landmarks
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                predicted_buffer.append(predicted_character)
                alphabet_label.configure(text=f"Alphabet Predicted: {predicted_character}")

                # Draw bounding box and predicted character on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(
                    frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA
                )

        else:
            # If no hand landmarks are detected, add the most common predicted character to the word
            if predicted_buffer:
                most_common = Counter(predicted_buffer).most_common(1)
                most_common_alphabet = most_common[0][0]
                word += most_common_alphabet
                word_label.configure(text=f"Word: {word}")
                predicted_buffer = []

        # Update the frame in the UI if not paused
        if not paused:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(image=frame_pil)  # Update the Tkinter image object
            frame_label.configure(image=frame_tk)
    except Exception as e:
        print("An error occurred:", e)
        pass

    root.after(100, update_ui)  # Schedule the next UI update

# Function bindings for various keyboard events
def reset_word_bind(event):
    reset_word()

def reset_sentence_bind(event):
    reset_sentence()

def toggle_pause_resume_bind(event):
    toggle_pause_resume()

def reset_word():
    """Reset the currently formed word."""
    global word, predicted_buffer
    word = ""
    predicted_buffer = []
    word_label.configure(text="Word:")

def reset_sentence():
    """Reset the entire sentence."""
    global sentence, incrementor
    sentence = ""
    sentence_label.configure(text="Sentence:")
    incrementor = 0

def toggle_pause_resume():
    """Toggle between pausing and resuming the video feed."""
    global paused
    paused = not paused
    if paused:
        pause_resume_button.configure(text="Resume (P)")
    else:
        pause_resume_button.configure(text="Pause (P)")

def append_word_to_sentence(event):
    """Append the current word to the sentence and reset the word."""
    global word, sentence, incrementor
    if sentence == "":
        if incrementor != 0 and len(word) != 1:
            sentence += word.lower() + " "
        elif len(word) == 1:
            sentence += word + " "
        else:
            first_word = word[0] + word[1:].lower()
            sentence += first_word + " "
    else:
        if incrementor != 0 and len(word) != 1:
            sentence += " " + word.lower() + " "
        elif len(word) == 1:
            sentence += " " + word + " "
        else:
            first_word = word[0] + word[1:].lower()
            sentence += " " + first_word + " "
    sentence_label.configure(text=f"Sentence: {sentence}")
    word = ""
    word_label.configure(text="Word:")
    incrementor += 1

def toggle_fullscreen(event):
    """Toggle between fullscreen and windowed mode."""
    global is_fullscreen
    is_fullscreen = not is_fullscreen
    root.attributes('-fullscreen', is_fullscreen)

def speaker(event):
    """Convert the sentence to speech using the pyttsx3 library."""
    global sentence
    engine = pyttsx3.init()
    text = sentence
    engine.say(text)
    engine.runAndWait()

def delete_last_letter(event):
    """Delete the last letter from the current word."""
    global word
    if len(word) > 0:
        word = word[:-1]
        word_label.configure(text=f"Word: {word}")

def delete_last_word(event):
    """Delete the last word from the sentence."""
    global sentence
    words = sentence.split()
    if len(words) > 0:
        sentence = ' '.join(words[:-1])  # Remove the last word
        sentence_label.configure(text=f"Sentence: {sentence}")

# Initialize the main window using customtkinter
root = customtkinter.CTk()
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")
is_fullscreen = True
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.title("Sign Recognition")
root.attributes('-fullscreen', is_fullscreen)
root.geometry(f"{screen_width-50}x{screen_height}")

# Create the main frame and divide it into left and right sections
main_frame = customtkinter.CTkFrame(master=root)
main_frame.pack(padx=50, pady=40, fill="both", expand=True)

left_frame = tk.Frame(main_frame)
left_frame.grid(row=0, column=0, sticky="nsew", padx=(50, 0), pady=(100, 0))
left_frame.configure(bg='#2b2b2b')

right_frame = tk.Frame(main_frame)
right_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 50), pady=(100, 0))
right_frame.configure(bg='#2b2b2b')

# Configure column proportions
main_frame.columnconfigure(1, weight=25)

# Label to display the webcam feed
frame_label = tk.Label(left_frame)
frame_label.grid(row=0, column=0, padx=0, pady=0)

# Label to display the predicted alphabet
alphabet_label = customtkinter.CTkLabel(
    master=right_frame,
    text="Alphabet Predicted:",
    font=("Arial", 50),
    text_color="#FFFFFF"
)
alphabet_label.grid(row=1, column=0, pady=20, padx=20, sticky="w")

# Label to display the currently forming word
word_label = customtkinter.CTkLabel(
    master=right_frame,
    text="Word:",
    font=("Arial", 50),
    text_color="#FFFFFF"
)
word_label.grid(row=2, column=0, pady=20, padx=20, sticky="w")

# Label to display the current sentence
sentence_label = customtkinter.CTkLabel(
    master=right_frame,
    text="Sentence:",
    font=("Arial", 50),
    text_color="#FFFFFF"
)
sentence_label.grid(row=3, column=0, pady=20, padx=20, sticky="w")

# Buttons for pausing, clearing word, and clearing sentence
pause_resume_button = customtkinter.CTkButton(
    master=right_frame,
    text="Pause (P)",
    font=("Arial", 30),
    text_color="#FFFFFF",
    command=toggle_pause_resume
)
pause_resume_button.grid(row=4, column=0, pady=10, padx=20, sticky="n")

clear_word_button = customtkinter.CTkButton(
    master=right_frame,
    text="Clear Word",
    font=("Arial", 30),
    text_color="#FFFFFF",
    command=reset_word
)
clear_word_button.grid(row=5, column=0, pady=10, padx=20, sticky="n")

clear_sentence_button = customtkinter.CTkButton(
    master=right_frame,
    text="Clear Sentence",
    font=("Arial", 30),
    text_color="#FFFFFF",
    command=reset_sentence
)
clear_sentence_button.grid(row=6, column=0, pady=10, padx=20, sticky="n")

# Bind keyboard shortcuts to corresponding functions
root.bind("<Return>", append_word_to_sentence)
root.bind("<F11>", toggle_fullscreen)
root.bind("<F5>", reset_word_bind)
root.bind("<F8>", reset_sentence_bind)
root.bind("<space>", toggle_pause_resume_bind)
root.bind("S", speaker)
root.bind("<BackSpace>", delete_last_letter)
root.bind("<Delete>", delete_last_word)

# Start the UI update loop
update_ui()

# Start the Tkinter main loop
root.mainloop()
