import cv2 as cv
import numpy as np
import os

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
DIR = r"C:\Users\sudha\OneDrive\Documents\Coding In Vs Code\Opencv\1Face Recognizer\Images"
people = []

for i in os.listdir(DIR):
    people.append(i)
print(people)


face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img=cv.imread(r"C:\Users\sudha\OneDrive\Documents\Coding In Vs Code\Opencv\1Face Recognizer\Images\pen\9.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)


# detect face 
face_rect=haar_cascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in face_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)