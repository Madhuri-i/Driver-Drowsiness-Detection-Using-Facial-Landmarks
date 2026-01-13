from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import time

# Initialize mixer and load alert sound
mixer.init()
mixer.music.load("music.wav")
mixer.music.set_volume(0.5)

# Calculate Eye Aspect Ratio (EAR) for eye openness detection
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Parameters for detection
thresh = 0.0          # EAR threshold (set after calibration)
frame_check = 20      # Number of consecutive frames below threshold before alert

# Load face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Get indexes for left and right eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

# -------- Calibration Phase --------
# Collect EAR values to compute baseline threshold
calibration_frames = []
calibration_duration = 5
start_time = time.time()

while time.time() - start_time < calibration_duration:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    cv2.putText(frame, "Look at the camera for calibration...", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        calibration_frames.append(ear)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

# Set threshold as 70% of baseline EAR or use default if calibration fails
if calibration_frames:
    baseline_ear = sum(calibration_frames) / len(calibration_frames)
    thresh = baseline_ear * 0.7
    print(f"Calibration complete. Baseline EAR: {baseline_ear:.3f}, Threshold: {thresh:.3f}")
else:
    print("Calibration failed. Using default threshold.")
    thresh = 0.25

# Variables for drowsiness detection
flag = 0
alert_active = False
drowsiness_start_time = None
face_rect_color = (0, 255, 0)  # Green rectangle initially

# -------- Real-time Monitoring --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    if len(subjects) == 0:
        cv2.putText(frame, "Face Not Found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not mixer.music.get_busy():
            mixer.music.play()  # Alert for no face detected
        flag = 0
        drowsiness_start_time = None
        face_rect_color = (0, 255, 0)

    else:
        if mixer.music.get_busy():
            mixer.music.stop()  # Stop alert if face detected

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            (x, y, w, h) = face_utils.rect_to_bb(subject)

            if ear < thresh:
                if drowsiness_start_time is None:
                    drowsiness_start_time = time.time()

                flag += 1

                if flag >= frame_check:
                    alert_active = True
                    face_rect_color = (0, 0, 255)  # Red rectangle
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not mixer.music.get_busy():
                        mixer.music.play()
            else:
                flag = 0
                alert_active = False
                drowsiness_start_time = None
                face_rect_color = (0, 255, 0)
                if mixer.music.get_busy():
                    mixer.music.stop()

            cv2.rectangle(frame, (x, y), (x + w, y + h), face_rect_color, 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
