import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Indices for left and right eyes (6 points each)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def get_landmarks(frame):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = []

    # Extract eye landmarks
    for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
        lm = face_landmarks.landmark[idx]
        h, w, _ = frame.shape
        x, y = int(lm.x * w), int(lm.y * h)
        landmarks.append((x, y))

    return landmarks[:6], landmarks[6:]  # (left_eye, right_eye)
