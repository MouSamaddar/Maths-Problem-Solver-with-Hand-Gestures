import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.set_page_config(page_title="Gesture-Based Math Solver", layout="wide")

# Setup MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def euclidean_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def detectGesture(hand1_data, hand2_data):
    (hand1, label1), (hand2, label2) = hand1_data, hand2_data
    f1 = count_fingers(hand1, label1)
    f2 = count_fingers(hand2, label2)
    dist = euclidean_distance(hand1.landmark[8], hand2.landmark[8])
    if f1==1 and f2==1:
        if dist < 0.06:
            return "exit"
        return "+"
    elif (f1==1 and f2==2) or (f1==2 and f2==1):
        return "-"
    elif (f1==1 and f2==3) or (f1==3 and f2==1):
        return "*"
    elif (f1==1 and f2==4) or (f1==4 and f2==1):
        return "/"
    elif (f1==2 and f2==2):
        return "del"
    elif (f1==1 and f2==5) or (f1==5 and f2==1):
        return "6"
    elif (f1==2 and f2==5) or (f1==5 and f2==2):
        return "7"
    elif (f1==3 and f2==5) or (f1==5 and f2==3):
        return "8"
    elif (f1==4 and f2==5) or (f1==5 and f2==4):
        return "9"
    elif f1==0 and f2==0:
        return "="
    elif f1==5 and f2==5:
        return "clear"
    return None

def count_fingers(hand_landmarks, label):
    tip_ids = [4,8,12,16,20]
    fingers = []
    if label=="Left":
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0]-1].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x else 0)
    for ids in range(1,5):
        fingers.append(1 if hand_landmarks.landmark[tip_ids[ids]].y < hand_landmarks.landmark[tip_ids[ids]-2].y else 0)
    return fingers.count(1)

# App State
st.title("🖐️ Gesture-Based Math Solver")
run = st.checkbox("Start Camera")

expression = st.empty()
result = st.empty()

last_update_time = 0
delay = 1.25
expr = ""
res = ""

if run:
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not detected.")
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        current_time = time.time()
        hand_data = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_handedness.classification[0].label
                hand_data.append((hand_landmarks, label))
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if len(hand_data) == 1:
                hand_landmarks, label = hand_data[0]
                fingers_up = count_fingers(hand_landmarks, label)
                if fingers_up in [0,1,2,3,4,5] and current_time - last_update_time > delay:
                    expr += str(fingers_up)
                    last_update_time = current_time

            elif len(hand_data) == 2:
                gesture = detectGesture(hand_data[0], hand_data[1])

                if gesture == "clear":
                    expr = ""
                    res = ""

                elif gesture == "exit":
                    st.warning("Exiting...")
                    break

                elif gesture and current_time - last_update_time > delay:
                    if gesture == "del":
                        expr = expr[:-1]
                    elif gesture == "=":
                        try:
                            res = str(eval(expr))
                        except:
                            res = "Error"
                    else:
                        expr += gesture
                    last_update_time = current_time

        # Display text
        cv2.putText(frame, f'Expr: {expr}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(frame, f'Res: {res}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
