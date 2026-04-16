import streamlit as st
import cv2
import dlib
import imutils
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import pygame
import os
import time

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Drowsiness Detection", layout="wide")

# ------------------- CUSTOM UI -------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

h1, h2, h3 {
    color: #3b82f6;
}

[data-testid="stSidebar"] {
    background-color: #020617;
}

.metric-card {
    background: #1e293b;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 0px 12px rgba(59,130,246,0.4);
}

.alert-box {
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown("""
<h1 style='text-align: center;'>🚨 AI Drowsiness Detection </h1>
<p style='text-align: center; color: #94a3b8;'>
Real-time fatigue monitoring system using Computer Vision
</p>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.markdown("---")

EYE_AR_THRESH = st.sidebar.slider("EAR Threshold", 0.15, 0.35, 0.25)
EYE_AR_CONSEC_FRAMES = st.sidebar.slider("Frame Sensitivity", 10, 40, 20)

start = st.sidebar.button("▶ Start Detection")
stop = st.sidebar.button("⏹ Stop Detection")

# ------------------- AUDIO -------------------
pygame.mixer.init()
ALARM_PATH = "assets/alarm.wav"

if not os.path.exists(ALARM_PATH):
    st.error("❌ Alarm file missing!")
    st.stop()

pygame.mixer.music.load(ALARM_PATH)

# ------------------- FUNCTION -------------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ------------------- LOAD MODELS -------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# ------------------- LAYOUT -------------------
col1, col2 = st.columns([3, 1])

frame_placeholder = col1.empty()
status_box = col2.empty()
ear_box = col2.empty()
alert_box = col2.empty()

# ------------------- STATE -------------------
if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False
    pygame.mixer.music.stop()

# ------------------- MAIN LOOP -------------------
if st.session_state.run:

    cap = cv2.VideoCapture(0)
    counter = 0
    alarm_on = False

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera not working")
            break

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        subjects = detector(gray, 0)

        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eyes
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0,255,0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0,255,0), 1)

            # EAR text
            cv2.putText(frame, f"EAR: {ear:.2f}", (400,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # ---------------- ALERT ----------------
            if ear < EYE_AR_THRESH:
                counter += 1

                if counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "🚨 DROWSY!", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                    if not alarm_on:
                        pygame.mixer.music.play(-1)
                        alarm_on = True

                    alert_box.markdown("""
                    <div class="alert-box" style="background:#ef4444;">
                        🚨 DROWSINESS DETECTED!
                    </div>
                    """, unsafe_allow_html=True)
            else:
                counter = 0

                alert_box.markdown("""
                <div class="alert-box" style="background:#22c55e;">
                    ✅ ACTIVE
                </div>
                """, unsafe_allow_html=True)

                if alarm_on:
                    pygame.mixer.music.stop()
                    alarm_on = False

            # EAR card
            ear_box.markdown(f"""
            <div class="metric-card">
                <h3>👁 EAR</h3>
                <h2>{ear:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        status_box.markdown("""
        <div class="metric-card">
            🟢 Camera Running
        </div>
        """, unsafe_allow_html=True)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)

        time.sleep(0.03)

    cap.release()
    pygame.mixer.music.stop()
    status_box.markdown("""
    <div class="metric-card">
        🛑 Stopped
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👉 Click 'Start Detection' from sidebar")