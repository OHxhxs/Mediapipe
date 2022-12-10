import cv2
import mediapipe as mp
import numpy as np
import math


mp_drawing = mp.solutions.drawing_utils     # 관절 드로잉
mp_pose = mp.solutions.pose         # 미디어파이프 솔루션


# 아크탄젠트 2를 사용하여 세점 사이의 각도 계산
def tp_angle(a,b,c):
    a = np.array(a)     # 첫번째 포인트, first_point
    b = np.array(b)     # 두번째 포인트, middle_point
    c = np.array(c)     # 세번째 포인트, end_point

    radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0/ np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,           # 최소 detection하는 confidence
    min_tracking_confidence=0.5) as pose:   # 최소 tracking하는 confidence

  while cap.isOpened():

    # frame = 캡처된 이미지
    _, frame = cap.read()

    # BGR로 되있기 때문에 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 이미지에 수정하는 것을 금지시킴
    image.flags.writeable = False

    # landmark 추출
    results = pose.process(frame)

    # 이미지 수정하는 것을 다시 가능하게 만듦
    image.flags.writeable = True

    # 이미지를 다시 BGR로 바꿔줌
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 이미지에 그리기
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 50, 230), thickness=2, circle_radius=2)
    )

    try:
        # 원하는 좌표 뽑아오기
        landmarks = results.pose_landmarks.landmark

        # 어깨값 가져오기
        # 위는 x값
        # 밑은 y값
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        print("shoulder(x,y):", shoulder)

        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        print("elbow(x,y):", elbow)

        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        print("wrist(x,y):", wrist)

        # 세점 사이의 각 구하기
        angle = tp_angle(shoulder, elbow, wrist)

        cv2.putText(image, str(round(angle,2)), tuple(np.multiply(elbow,[640.480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0), 4)

    except:
        pass


    # 보기좋게 좌우반전까지.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

    # 정지시키는 코드
    if cv2.waitKey(5) & 0xFF == 27:

    # if cv2.waitKey(10) & OxFF == ord('q'):
      break
cap.release()