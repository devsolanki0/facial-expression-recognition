import cv2
import numpy as np
import tensorflow as tf
import time

# ==========================================
# 1. Load trained model
# ==========================================

MODEL_PATH = "models/best_facial_expression_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# Expression labels
class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ==========================================
# 2. Load OpenCV face detector
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# 3. Start webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# FPS variables
prev_time = 0

print("Camera started.")
print("Press 'q' to quit.")

# ==========================================
# 4. Real-time loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # ======================================
    # Process every detected face
    # ======================================

    for (x, y, w, h) in faces:

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize to model input
        face = cv2.resize(face, (48, 48))

        # Normalize
        face = face.astype("float32") / 255.0

        # Add batch + channel dimensions
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # ==================================
        # Prediction
        # ==================================

        predictions = model.predict(face, verbose=0)[0]

        predicted_index = np.argmax(predictions)

        expression = class_names[predicted_index]

        confidence = predictions[predicted_index] * 100

        # ==================================
        # Draw face rectangle
        # ==================================

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # ==================================
        # Display expression
        # ==================================

        text = f"{expression}: {confidence:.1f}%"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # ======================================
    # Calculate FPS
    # ======================================

    current_time = time.time()

    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # ======================================
    # Display camera
    # ======================================

    cv2.imshow(
        "FaceSense AI - Real-Time Expression Recognition",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# Cleanup
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Camera stopped.")