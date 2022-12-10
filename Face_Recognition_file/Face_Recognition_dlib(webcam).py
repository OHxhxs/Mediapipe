'''

웹캠에서 실시간으로 face_recognition으로 detecting
128개의 점

'''


import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# 이미지의 이름들을 className으로 사용, 이미지들을 따로 images에 저장
path = 'img_attedance'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# 이미지들을 인코딩 해서 List에 담는 것.
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

# # 현재 날짜와 시간 csv에 저장
# def markAttendance(name):
#     with open('Attendance.csv','r+') as f :
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

# 웹캠에서 사용S
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # 속도를 위해 resize
    imgS = cv2.resize(img, (0,0),None, 0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 현재 프레임에서 얼굴 좌표 찾는것
    facesCurFrame = face_recognition.face_locations(imgS)
    
    # 인코딩
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    print(encodeCurFrame)

    # 비교하기
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        print(encodeFace)
        print(type(encodeFace))

        # encodeListKnown에는 3개의 이미지가 있기 때문에 3개가 있는 리스트로 반환됨
        # 그 중에 faceDis가 가장 낮은 것이 가장 유사한 사진
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)

        # 가장 낮은 faceDis의 인덱스 찾기
        matchIndex = np.argmin(faceDis)

        # 밑에 처럼 안한 이유가 지금 test 중에서 내 얼굴과 유사한 사진이
        # 없기에 print(name)을 했을떄 name이 안나옴
        # 실전에서는 밑에처럼 적을 것.
        if matches[matchIndex]:
        # if classNames[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc

            # 나는 왜 이거 안해도 잘되지?
            # y1, x2, y2, x1 = y1*4, x2*4 ,y2*4 ,x1*4

            # bbox그리기
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,25,0),2)

            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)

            cv2.putText(img, name,(x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255), 2)

            # markAttendance(name)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(10) == 27:
        break
