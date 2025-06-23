import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2
import time

st.set_page_config(page_title="Math Solver with Hand Gestures")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Utility: count raised fingers
def count_fingers(hand_landmarks, label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    if label == "Left":
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0]-1].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x else 0)
    for i in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i]-2].y else 0)
    return fingers.count(1)

# Detect gesture based on fingers and position
def detectGesture(hand1_data, hand2_data):
    (hand1, label1), (hand2, label2) = hand1_data, hand2_data
    f1 = count_fingers(hand1, label1)
    f2 = count_fingers(hand2, label2)
    dist = np.sqrt((hand1.landmark[8].x - hand2.landmark[8].x) ** 2 + (hand1.landmark[8].y - hand2.landmark[8].y) ** 2)
    if f1 == 1 and f2 == 1:
        if dist < 0.06:
            return "exit"
        return "+"
    elif (f1 == 1 and f2 == 2) or (f1 == 2 and f2 == 1):
        return "-"
    elif (f1 == 1 and f2 == 3) or (f1 == 3 and f2 == 1):
        return "*"
    elif (f1 == 1 and f2 == 4) or (f1 == 4 and f2 == 1):
        return "/"
    elif (f1 == 2 and f2 == 2):
        return "del"
    elif f1 == 0 and f2 == 0:
        return "="
    elif f1 == 5 and f2 == 5:
        return "clear"
    return None

# Streamlit UI
st.title("🖐️ Math Solver with Hand Gestures")
st.markdown("Use hand gestures to build and solve math expressions!")

if 'expression' not in st.session_state:
    st.session_state.expression = ""
if 'result' not in st.session_state:
    st.session_state.result = ""

img_file = st.camera_input("Show your hands to the camera")

if img_file is not None:
    img = Image.open(img_file)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    hand_data = []
    if results.multi_hand_landmarks and results.multi_handedness:
        for landmark, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            hand_data.append((landmark, label))
            mp_drawing.draw_landmarks(img, landmark, mp_hands.HAND_CONNECTIONS)

        if len(hand_data) == 1:
            count = count_fingers(hand_data[0][0], hand_data[0][1])
            st.session_state.expression += str(count)
        elif len(hand_data) == 2:
            gesture = detectGesture(hand_data[0], hand_data[1])
            if gesture == "exit":
                st.warning("Exit gesture detected.")
            elif gesture == "clear":
                st.session_state.expression = ""
                st.session_state.result = ""
            elif gesture == "del":
                st.session_state.expression = st.session_state.expression[:-1]
            elif gesture == "=":
                try:
                    st.session_state.result = str(eval(st.session_state.expression))
                except:
                    st.session_state.result = "Error"
            elif gesture:
                st.session_state.expression += gesture

    st.image(img, channels="BGR")

st.markdown(f"### 🧮 Expression: `{st.session_state.expression}`")
st.markdown(f"### ✅ Result: `{st.session_state.result}`")
