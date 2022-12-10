'''
pip install mediapipe

사이트 참조 : https://puleugo.tistory.com/5

얼굴을 찾고 얼굴 그물망을 그려줌

'''

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# # 이미지 파일의 경우을 사용하세요.:
# IMAGE_FILES = ['test.jpg']
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# with mp_face_mesh.FaceMesh(
#         static_image_mode=True,
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5) as face_mesh:
#     for idx, file in enumerate(IMAGE_FILES):
#         image = cv2.imread(file)
#         # 작업 전에 BGR 이미지를 RGB로 변환합니다.
#         results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#         # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
#         if not results.multi_face_landmarks:
#             continue
#         annotated_image = image.copy()
#         for face_landmarks in results.multi_face_landmarks:
#             print('face_landmarks:', face_landmarks)
#             mp_drawing.draw_landmarks(
#                 image=annotated_image,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp_drawing_styles
#                 .get_default_face_mesh_tesselation_style())
#             mp_drawing.draw_landmarks(
#                 image=annotated_image,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_CONTOURS,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp_drawing_styles
#                 .get_default_face_mesh_contours_style())
#             mp_drawing.draw_landmarks(
#                 image=annotated_image,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_IRISES,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp_drawing_styles
#                 .get_default_face_mesh_iris_connections_style())
#         cv2.imwrite('./Face_Mesh_img_File/' +
#                     str(idx) + '.jpg', annotated_image)

# 웹캠, 영상 파일의 경우 이것을 사용하세요.:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("웹캠을 찾을 수 없습니다.")
            # 비디오 파일의 경우 'continue'를 사용하시고, 웹캠에 경우에는 'break'를 사용하세요
            continue

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # 이미지 위에 얼굴 그물망 주석을 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                print(face_landmarks)

                # 양쪽 눈썹, 양쪽 눈, 입, 얼굴 형, 입, 눈동자 빼고 나머지 좌표값들
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                print(mp_face_mesh.FACEMESH_TESSELATION)

                # 눈과 눈썹, 얼굴형, 입 그려주기
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                print(mp_face_mesh.FACEMESH_CONTOURS)

                # 눈동자 인식
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                print(mp_face_mesh.FACEMESH_IRISES)

        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Face Mesh(Puleugo)', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()