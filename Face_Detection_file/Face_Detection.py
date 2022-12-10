'''
pip install mediapipe

사이트 참조 : https://puleugo.tistory.com/4

얼굴에 bbox를 치고 6개의 점을 찾음
눈 : 2
코 : 1
입 : 1
귀 : 2

'''

import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 이미지 파일의 경우 이것을 사용하세요:
# IMAGE_FILES = []
IMAGE_FILES = ['test.jpg']
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:

  for idx, file in enumerate(IMAGE_FILES):
    print(idx, file)
    image = cv2.imread(file)
    # 작업 전에 BGR 이미지를 RGB로 변환합니다.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	# 이미지를 출력하고 그 위에 얼굴 박스를 그립니다.
    if not results.detections:
      continue
    annotated_image = image.copy()

    for detection in results.detections:
      print('Nose tip:')
      print(mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
    cv2.imwrite('./Face_Detection_img_file/' + str(idx) + '.jpg', annotated_image)



# # 웹캠, 영상 파일의 경우 사용:
#
# # 웹캠 키기
# cap = cv2.VideoCapture(0)
# with mp_face_detection.FaceDetection(
#     model_selection=0, min_detection_confidence=0.5) as face_detection:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("웹캠을 찾을 수 없습니다.")
#       # 비디오 파일의 경우 'continue', 웹캠에 경우에는 'break'를 사용하세요.
#       continue
# 	# 보기 편하기 위해 이미지를 좌우 반전, BGR 이미지를 RGB로 변환합니다.
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#
# 	# 성능을 향상시키려면 이미지를 작성 여부를 False로 설정
#     image.flags.writeable = False
#     results = face_detection.process(image)
#
#     # 영상에 얼굴 감지 주석 그리기 기본값 : True.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.detections:
#       for detection in results.detections:
#         print(detection)
#         mp_drawing.draw_detection(image, detection)
#     cv2.imshow('MediaPipe Face Detection', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()

