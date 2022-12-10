'''
< Face Recognition >

- 참고
블로그 : https://wiserloner.tistory.com/1198
유튜브 : https://www.youtube.com/watch?v=sz25xxF_AVE&t=28s

face recognition : https://velog.io/@wbsl0427/facerecognition%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EC%96%BC%EA%B5%B4%EC%9D%B8%EC%8B%9D-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8-%EB%A7%8C%EB%93%A4%EA%B8%B0

pip install face-recognition
pip install cmake
pip isntall dlib
'''

import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('./img_file/musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('./img_file/musk_test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# 얼굴 찾기

# 원본 이미지 얼굴 bbox 좌표값
faceLoc = face_recognition.face_locations(imgElon)[0]

# 이를 encoding하여 특정 128개의 값으로 만들어내는 것.
encodeElon = face_recognition.face_encodings(imgElon)[0]
# print(encodeElon)
# bbox 그리기
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]),(255,0,255), 2)

# Test이미지
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]),(255,0,255), 2)

# 원본 이미지와 테스트 이미지가 비슷한지 테스트
results = face_recognition.compare_faces([encodeElon], encodeTest)
# print(results)

faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)

# bbox위에 글자 입력
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
