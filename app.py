# pip install torch torchvision transformers opencv-python


import cv2
import pygame
import threading
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Initialize Pygame mixer
pygame.mixer.init()
sound_file = 'female_sound.mp3'  # Path to the MP3 file
playing = False  # Flag to control playback

def play_sound(sound_path):
    global playing
    pygame.mixer.music.load(sound_path)
    while playing:
        pygame.mixer.music.play(-1)  # Play the sound in a loop
        while pygame.mixer.music.get_busy():  # Wait until the sound finishes
            pygame.time.Clock().tick(10)  # Sleep for a while to prevent busy-waiting

# Use the default camera (0 for the first camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(rgb_frame)

    # Check if any pose landmarks are detected
    if results.pose_landmarks:
        # Draw the pose landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Start playing sound if not already playing
        if not playing:
            playing = True
            sound_thread = threading.Thread(target=play_sound, args=(sound_file,))
            sound_thread.start()
    else:
        # Stop the music if no poses are detected
        if playing:
            playing = False
            pygame.mixer.music.stop()  # Stop the music when no poses are detected

    # Display the frame
    cv2.imshow("Pose Tracking", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
playing = False  # Stop the sound playback loop
pygame.mixer.music.stop()  # Stop music when exiting
cap.release()
cv2.destroyAllWindows()
