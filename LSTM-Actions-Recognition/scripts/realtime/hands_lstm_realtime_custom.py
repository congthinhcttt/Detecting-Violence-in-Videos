import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import h5py
import json

# ================= CAMERA =================
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)

# ================= MEDIAPIPE =================
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# ================= MODEL LOAD =================
custom_objects = {
    "Orthogonal": tf.keras.initializers.Orthogonal
}

with h5py.File("lstm-hand-grasping.h5", "r") as f:
    model_config = json.loads(f.attrs.get("model_config"))

    for layer in model_config["config"]["layers"]:
        if "time_major" in layer["config"]:
            del layer["config"]["time_major"]

    model_json = json.dumps(model_config)
    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)

    weights_group = f["model_weights"]
    for layer in model.layers:
        if layer.name in weights_group:
            weight_names = weights_group[layer.name].attrs["weight_names"]
            weights = [weights_group[layer.name][w] for w in weight_names]
            layer.set_weights(weights)

print("✅ Model loaded successfully")

# ================= ACTION DEFINITIONS =================
ACTIONS = ["neutral", "resting", "carrying", "cupping"]

NEUTRAL_ACTIONS = ["neutral", "resting"]
GRASP_ACTIONS = ["carrying", "cupping"]

label = "DETECTING..."
lm_list = []

SEQUENCE_LENGTH = 20
CONF_THRESHOLD = 0.6

# ================= FUNCTIONS =================
def make_landmark_timestep(results):
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks


def draw_landmarks(results, frame):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
    return frame


def draw_bbox_and_label(frame, results, label):
    if not results.multi_hand_landmarks:
        return frame

    h, w, _ = frame.shape
    for hand_landmarks in results.multi_hand_landmarks:
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]

        x_min, x_max = int(min(xs) * w), int(max(xs) * w)
        y_min, y_max = int(min(ys) * h), int(max(ys) * h)

        color = (0, 0, 255) if label == "GRASPED" else (0, 255, 0)
        thickness = 3 if label == "GRASPED" else 2

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

        cv2.putText(
            frame,
            f"Action: {label}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA
        )
    return frame


def detect_action(model, lm_sequence):
    global label

    lm_sequence = np.array(lm_sequence)
    lm_sequence = np.expand_dims(lm_sequence, axis=0)

    result = model.predict(lm_sequence, verbose=0)
    confidence = np.max(result)
    action_index = np.argmax(result)
    action = ACTIONS[action_index]

    print(f"Prediction: {action} ({confidence*100:.2f}%)")

    if confidence < CONF_THRESHOLD:
        label = "DETECTING..."
    elif action in NEUTRAL_ACTIONS:
        label = "NOT GRASPED"
    elif action in GRASP_ACTIONS:
        label = "GRASPED"
    else:
        label = "UNKNOWN"


# ================= WINDOW =================
cv2.namedWindow("Violence Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Violence Detection", 1200, 900)

warm_up_frames = 60
frame_count = 0

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame_count += 1

    if frame_count > warm_up_frames and results.multi_hand_landmarks:
        landmarks = make_landmark_timestep(results)
        lm_list.append(landmarks)

        if len(lm_list) == SEQUENCE_LENGTH:
            threading.Thread(
                target=detect_action,
                args=(model, lm_list)
            ).start()
            lm_list = []

    frame = draw_landmarks(results, frame)
    frame = draw_bbox_and_label(frame, results, label)

    cv2.imshow("Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
