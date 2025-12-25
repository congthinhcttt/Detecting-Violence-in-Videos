import cv2
import mediapipe as mp
import pandas as pd

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Cài đặt độ phân giải cho webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  # Chiều rộng
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  # Chiều cao

# Khởi tạo Mediapipe Pose với hỗ trợ nhiều người
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "neutral"
no_of_frames = 1000
i = 0

# Hàm chuyển đổi pose landmarks thành danh sách tọa độ và độ hiển thị
def make_landmark_timestep(landmarks):
    c_lm = []
    for id, lm in enumerate(landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

# Hàm vẽ các landmarks trên hình ảnh
def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

# Hàm kiểm tra hành động "punch" bằng vai
def is_punch_detected(landmarks):
    left_shoulder = landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST]

    if (left_shoulder.visibility > 0.3 and
            left_elbow.visibility > 0.3 and
            left_wrist.visibility > 0.3):
        if (left_wrist.y < left_elbow.y - 0.1):
            return True
    return False

# Hàm kiểm tra hành động "kick" bằng cả hai chân
def is_kick_detected(landmarks):
    left_hip = landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP]
    left_knee = landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE]
    left_ankle = landmarks.landmark[mpPose.PoseLandmark.LEFT_ANKLE]

    right_hip = landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP]
    right_knee = landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE]
    right_ankle = landmarks.landmark[mpPose.PoseLandmark.RIGHT_ANKLE]

    left_kick = (left_hip.visibility > 0.3 and left_knee.visibility > 0.3 and left_ankle.visibility > 0.3 and
                 left_ankle.y < left_knee.y)
    right_kick = (right_hip.visibility > 0.3 and right_knee.visibility > 0.3 and right_ankle.visibility > 0.3 and
                  right_ankle.y < right_knee.y)

    return left_kick or right_kick

# Hàm kiểm tra hành động "bóp cổ"
def is_choke_detected(landmarks):
    left_shoulder = landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER]
    neck = landmarks.landmark[mpPose.PoseLandmark.NOSE]
    left_wrist = landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST]

    # Kiểm tra xem hai cổ tay có nằm gần nhau và ở vị trí gần cổ
    if (left_wrist.visibility > 0.3 and right_wrist.visibility > 0.3 and
            left_shoulder.visibility > 0.3 and right_shoulder.visibility > 0.3):
        # Nếu cổ tay ở gần cổ
        if (left_wrist.y < neck.y + 0.1 and right_wrist.y < neck.y + 0.1 and
                abs(left_wrist.x - right_wrist.x) < 0.15):  # Thay đổi ngưỡng tùy theo độ rộng
            return True
    return False


# Thiết lập cửa sổ hiển thị
cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

# Vòng lặp để xử lý từng khung hình
while len(lm_list) < no_of_frames:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame")
        break

    # Che đi timestamp (nếu có) bằng hình chữ nhật màu đen ở góc trên bên trái
    cv2.rectangle(frame, (0, 0), (200, 50), (0, 0, 0), -1)

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

    if results.pose_landmarks:
        lm = make_landmark_timestep(results.pose_landmarks)
        lm_list.append(lm)
        frame = draw_landmark_on_image(mpDraw, results, frame)

        # Kiểm tra và cập nhật nhãn label
        if is_punch_detected(results.pose_landmarks):
            label = "punch"
            cv2.putText(frame, "Punch Detected!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3, cv2.LINE_AA)
        elif is_kick_detected(results.pose_landmarks):
            label = "kick"
            cv2.putText(frame, "Kick Detected!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3, cv2.LINE_AA)

            # Ghi vào tệp kick.txt khi phát hiện hành động đá
            with open("kick.txt", "a") as f:
                f.write("Kick detected at frame {}\n".format(len(lm_list)))
        elif is_choke_detected(results.pose_landmarks):
            label = "choke"
            cv2.putText(frame, "Choke Detected!", (100, 300), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 0, 255), 3, cv2.LINE_AA)  # Màu tím cho bóp cổ

            # Ghi vào tệp choke.txt khi phát hiện hành động bóp cổ
            with open("choke.txt", "a") as f:
                f.write("Choke detected at frame {}\n".format(len(lm_list)))

        else:
            label = "neutral"
            cv2.putText(frame, "Neutral", (100, 400), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 255), 3, cv2.LINE_AA)

    # Thay đổi kích thước khung hình về 800x600
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Lưu danh sách landmarks thành tệp CSV
df = pd.DataFrame(lm_list)
df.to_csv(f"{label}_{i}.csv", index=False)
i += 1  # Tăng biến i để tạo tên tệp duy nhất cho lần sau
# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
