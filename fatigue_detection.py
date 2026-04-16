from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame
import os

# 🔊 Initialize pygame mixer (ONLY ONCE)
pygame.mixer.init()

# ✅ Correct path to alarm file
ALARM_PATH = "assets/alarm.wav"

# Check if file exists
if not os.path.exists(ALARM_PATH):
    print("❌ Alarm file not found! Check path:", ALARM_PATH)
    exit()

pygame.mixer.music.load(ALARM_PATH)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 🔧 Parameters
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

# 🧠 Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# 👁 Eye landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# 🎥 Webcam
cap = cv2.VideoCapture(0)

# Improve camera stability
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not working")
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    subjects = detector(gray, 0)

    for subject in subjects:
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # 👁 Extract eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # 👁 Draw contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 📊 Show EAR
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 🚨 Drowsiness logic
        if ear < EYE_AR_THRESH:
            counter += 1

            if counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "🚨 DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 🔊 Start alarm (only once)
                if not alarm_on:
                    pygame.mixer.music.play(-1)  # loop forever
                    alarm_on = True
        else:
            counter = 0

            # 🛑 Stop alarm
            if alarm_on:
                pygame.mixer.music.stop()
                alarm_on = False

    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# 🔚 Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()