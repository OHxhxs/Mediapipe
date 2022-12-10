import csv

import cv2
import mediapipe as mp
import numpy as np
import os

# pkl파일 불러오기 위해서 사용하기 위해
import joblib
import pandas as pd
model = joblib.load('face_rf.pkl')



mp_drawing = mp.solutions.drawing_utils # 관절드로잉
mp_holistic = mp.solutions.holistic # 미디어파이프 솔루션

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
        min_detection_confidence = 0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        _,frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 120), thickness=1, circle_radius=1),
                                  )
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=3, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 50, 230), thickness=3, circle_radius=2),
                                  )

        try:
            # face landmark값 가져오기

            face = results.face_landmarks.landmark
            face_list = []
            for temp in face:
                face_list.append([temp.x, temp.y, temp.z, temp.visibility])

            face_row = list(np.array(face_list).flatten())


            #### 모델 예측 ####
            X = pd.DataFrame([face_row])
            print(model.predict(X)[0])

            if os.path.isfile('coords.csv') == False:  # 만약 파일이 없으면

                landmarks = ['label']

                for temp in range(1, len(face) + 1):

                    # csv에 column명
                    # x1, y1, z1 이런식으로 들어감
                    landmarks += ['x{}'.format(temp), 'y{}'.format(temp), 'z{}'.format(temp), 'v{}'.format(temp)]

                # print(landmarks)

                with open('coords.csv', mode='w', newline='') as f:

                    csv_writer = csv.writer(f, delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(landmarks)
                    f.close()

            else:
                if cv2.waitKey(10) & 0xFF == ord('s'):
                    face_row.insert(0, 'happy')             # 0번째 column에 happy 집어넣기
                    with open('coords.csv', mode='a', newline='') as f: # 계속 추가할거기 때문에 mode='a'
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(face_row)
                        f.close()

                elif cv2.waitKey(10) & 0xFF == ord('d'):
                    face_row.insert(0, 'sad')             # 0번째 column에 sad 집어넣기
                    with open('coords.csv', mode='a', newline='') as f: # 계속 추가할거기 때문에 mode='a'
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(face_row)
                        f.close()


        except:
            pass


        cv2.imshow('holistic', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

