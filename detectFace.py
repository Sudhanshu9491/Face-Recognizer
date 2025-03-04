import os
import cv2 as cv
import numpy as np

DIR = r"C:\Users\sudha\OneDrive\Documents\Coding In Vs Code\Opencv\1Face Recognizer\Images"
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

people = []

for i in os.listdir(DIR):
    people.append(i)
print(people)

faceOfPerson=[]
nameOfPerson=[]

def create_train():
    for p in people:
        path=os.path.join(DIR,p)
        name=people.index(p)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5 )
            for (x,y,w,h) in face_rect:
                face_roi=gray[y:y+h,x:x+w]
                faceOfPerson.append(face_roi)
                nameOfPerson.append(name)

create_train()
# print(f'Length of faceOfPerson = {len(faceOfPerson)}')
# print(f'Length of nameOfPerson = {len(nameOfPerson)}')

print("--------------Training Done--------------")

faceOfPerson=np.array(faceOfPerson,dtype='object')
nameOfPerson=np.array(nameOfPerson)

face_recognizer=cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(faceOfPerson,nameOfPerson)
face_recognizer.save('face_trained.yml')
np.save('faceOfPerson.npy',faceOfPerson)
np.save('nameOfPerson.npy',nameOfPerson)