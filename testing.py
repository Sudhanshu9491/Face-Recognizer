import cv2 as cv
import numpy as np
import os

# Load Haar Cascade for face detection
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define dataset directory
DIR = r"C:\Users\sudha\OneDrive\Documents\Coding In Vs Code\Opencv\1Face Recognizer\Images"
people = [p for p in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, p))]
print("People:", people)

# Load trained face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
if not os.path.exists('face_trained.yml'):
    print("Error: Trained model file not found!")
    exit()
face_recognizer.read('face_trained.yml')

# Load and process the test image
img_path = r"C:\Users\sudha\OneDrive\Documents\Coding In Vs Code\Opencv\1Face Recognizer\Images\Ben\8.jpg"
img = cv.imread(img_path)
if img is None:
    print("Error: Could not read image!")
    exit()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect face
face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
if len(face_rect) == 0:
    print("No face detected!")
else:
    for (x, y, w, h) in face_rect:
        faces_roi = gray[y:y+h, x:x+w]
        faces_roi = cv.resize(faces_roi, (100, 100))  # Ensure size consistency
        
        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')
        
        # Display results
        text = f'{people[label]} ({int(confidence)})' if confidence < 100 else "Unknown"
        cv.putText(img, text, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
cv.destroyAllWindows()
