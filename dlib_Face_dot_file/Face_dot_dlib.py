'''
< dlib >
참고 : https://choiiis.github.io/machine-learning/detect-facial-landmarks-with-dlib-and-save-as-json/

- anaconda prompt 들어가서 실행
pip install cmake
conda install -c conda-forge dlib

학습된 모델 : http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 다운
7-zip 으로 bz2 파일 압축풀기

68개의 얼굴의 특징점을 잡아냄

'''

import dlib
import cv2
import numpy as np
import json
#
# # create list for landmarks
# ALL = list(range(0, 68))
# RIGHT_EYEBROW = list(range(17, 22))
# LEFT_EYEBROW = list(range(22, 27))
# RIGHT_EYE = list(range(36, 42))
# LEFT_EYE = list(range(42, 48))
# NOSE = list(range(27, 36))
# MOUTH_OUTLINE = list(range(48, 61))
# MOUTH_INNER = list(range(61, 68))
# JAWLINE = list(range(0, 17))
#
#
# create face detector, predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
#
# # 이미지
#
# image = cv2.imread('../test.jpg')
# img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Get faces (up-sampling=1)
# face_detector = detector(img_gray, 1)
# # the number of face detected
# print("The number of faces detected : {}".format(len(face_detector)))
#
# # loop as the number of face
# # one loop belong to one face
# for face in face_detector:
#     # face wrapped with rectangle
#     cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),
#                   (0, 0, 255), 3)
#
#     # make prediction and transform to numpy array
#     landmarks = predictor(image, face)  # 얼굴에서 68개 점 찾기
#
#     # create list to contain landmarks
#     landmark_list = []
#
#     # append (x, y) in landmark_list
#     for p in landmarks.parts():
#         landmark_list.append([p.x, p.y])
#         cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)
#
#     cv2.imwrite('./dlib_Face_dot_file/face_dlib_test.jpg',image)
#
#     with open("test.json", "w") as json_file:
#         key_val = [ALL, landmark_list]
#         landmark_dict = dict(zip(*key_val))
#         print(landmark_dict)
#         json_file.write(json.dumps(landmark_dict))
#         json_file.write('\n')

# 웹캠 이용
# create VideoCapture object (input the video)
# 0 for web camera
vid_in = cv2.VideoCapture(0)
# "---" for the video file
#vid_in = cv2.VideoCapture("baby_vid.mp4")

# capture the image in an infinite loop
# -> make it looks like a video
while True:
    # Get frame from video
    # get success : ret = True / fail : ret= False
    ret, image_o = vid_in.read()

   # resize the video
    image = cv2.resize(image_o, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces (up-sampling=1)
    face_detector = detector(img_gray, 1)
    # the number of face detected
    print("The number of faces detected : {}".format(len(face_detector)))

    # loop as the number of face
    # one loop belong to one face
    for face in face_detector:
        # face wrapped with rectangle
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),
                      (0, 0, 255), 3)

        # make prediction and transform to numpy array
        landmarks = predictor(image, face)  # 얼굴에서 68개 점 찾기

        #create list to contain landmarks
        landmark_list = []

        # append (x, y) in landmark_list
        for p in landmarks.parts():
            landmark_list.append([p.x, p.y])
            cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)


    cv2.imshow('result', image)

    # wait for keyboard input
    key = cv2.waitKey(1)

    # if esc,
    if key == 27:
        break


    # # json으로 잡는 것.
    # with open("test.json", "w") as json_file:
    #     key_val = [ALL, landmark_list]
    #     landmark_dict = dict(zip(*key_val))
    #     print(landmark_dict)
    #     json_file.write(json.dumps(landmark_dict))
    #     json_file.write('\n')

vid_in.release()