import cv2
import mediapipe as mp
import pandas as pd

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "neutral"
no_of_frames = 1000


def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame


i = 0
while i < no_of_frames:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

    if results.pose_landmarks:
        lm = make_landmark_timestep(results)
        lm_list.append(lm)
        i += 1  # Chỉ tăng i khi phát hiện được landmarks
        frame = draw_landmark_on_image(mpDraw, results, frame)
    else:
        print("No landmarks detected")

    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Lưu kết quả
df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt", index=False)

cap.release()
cv2.destroyAllWindows()