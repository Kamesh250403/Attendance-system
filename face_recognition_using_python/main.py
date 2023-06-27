import cv2
import numpy as np
import  face_recognition

##############################
import pandas as pd
import pymongo
from pymongo import MongoClient

##############################




import os
from datetime import datetime

cluster = MongoClient("mongodb+srv://a********ev:*****@cluster0.uo8dv.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")




path = 'Student list'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncondings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%y")
    path = "C:\\Users\\sanjeev\\PycharmProjects\\Face_rec\\Attendance\\" + timestampStr + '.csv'
    # path="C:\\Users\\sanjeev\\PycharmProjects\\Face_rec\\pymongo_test_insert.py"

    with open(path, 'a+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            drString = now.strftime("%d %B %y")


            post = {
                "name": name,
                "time": dtString,
                "date": drString
            }
            db = cluster["attendance"]
            collection = db["lists"]
            x = collection.find()

            alreadyIn = False
            for i in x:
                if post['name'] == i['name']:
                    alreadyIn = True
            if not alreadyIn:
                collection.insert_one(post)
                f.writelines(f'{name},{dtString},{drString}\n')


encodeListKnown = findEncondings(images)
print('Encoding Completed')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)



