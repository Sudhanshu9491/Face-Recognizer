import os
import cv2 as cv
import numpy as np

# Load Haar Cascade
detector = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define dataset directory
DIR = r"C:\Users\sudha\OneDrive\Documents\Coding In Vs Code\Opencv\1Face Recognizer\Images"
people = [p for p in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, p))]
print("People:", people)

# Initialize training data
faceOfPerson, nameOfPerson = [], []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv.imread(img_path)
            if img is None:
                continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv.resize(face_roi, (100, 100))  # Ensure uniform size
                faceOfPerson.append(face_roi)
                nameOfPerson.append(label)
                
create_train()

if len(faceOfPerson) == 0:
    print("Error: No faces found for training!")
    exit()

print("Training on", len(faceOfPerson), "faces")
faceOfPerson = np.array(faceOfPerson, dtype='object')
nameOfPerson = np.array(nameOfPerson, dtype=np.int32)

# Train face recognizer
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.train(faceOfPerson, nameOfPerson)
recognizer.save('face_trained.yml')
np.save('faceOfPerson.npy', faceOfPerson)
np.save('nameOfPerson.npy', nameOfPerson)
print("Training Completed!")
