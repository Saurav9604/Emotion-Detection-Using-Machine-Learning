import cv2
import numpy as np
import time
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Load the emotion labels (adjust based on your model's output)
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open and read the JSON model file
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

# Load the model from JSON string, passing the custom_objects argument
model = model_from_json(model_json, custom_objects={
    'Sequential': Sequential,
    'Conv2D': Conv2D,
    'MaxPooling2D': MaxPooling2D,
    'Dropout': Dropout,
    'Flatten': Flatten,
    'Dense': Dense
})

# Load the model weights
model.load_weights("facialemotionmodel.h5")

# Initialize the face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture (change source if using a video file instead of webcam)
cap = cv2.VideoCapture(0)

# Control the frame rate (e.g., predict every 5 frames)
frame_counter = 0
prediction_interval = 5  # Predict every 5 frames
last_predicted_emotion = "Neutral"  # Cache last emotion for skipped frames

# Main loop for real-time detection
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (face detection works better with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop and preprocess the face region for emotion detection
        face_region = gray[y:y + h, x:x + w]
        face_region_resized = cv2.resize(face_region, (48, 48))  # Model expects 48x48 input
        face_region_normalized = face_region_resized / 255.0  # Normalize pixel values
        face_region_input = np.reshape(face_region_normalized, (1, 48, 48, 1))

        # Only predict every `prediction_interval` frames
        if frame_counter % prediction_interval == 0:
            emotion_prediction = model.predict(face_region_input)
            max_index = np.argmax(emotion_prediction[0])  # Get index of highest probability
            confidence = emotion_prediction[0][max_index]  # Get confidence of prediction
            
            # Update last_predicted_emotion only if confidence > 60%
            if confidence > 0.6:
                last_predicted_emotion = emotion_labels[max_index]

        # Display the last predicted emotion on the frame
        cv2.putText(frame, last_predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Increment frame counter
    frame_counter += 1

    # Control frame rate (approx 30 FPS, ~33ms delay per frame)
    time.sleep(0.033)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
