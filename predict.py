"""
Step 3: Live prediction from your laptop camera.

Usage:
    python predict.py

Press Q to quit.
"""
import json

import cv2
import numpy as np
from tensorflow import keras

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.6  # below this, show "Not sure"

model = keras.models.load_model("model.keras")
with open("labels.json") as f:
    class_names = json.load(f)
print("Loaded classes:", class_names)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open camera.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Same preparation as training: RGB, 224x224, batch of 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, IMG_SIZE).astype("float32")
    probs = model(np.expand_dims(img, 0), training=False).numpy()[0]

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    if confidence >= CONFIDENCE_THRESHOLD:
        text, color = f"{class_names[idx]}  {confidence * 100:.0f}%", (0, 255, 0)
    else:
        text, color = "Not sure...", (0, 165, 255)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    # Small bar chart of all classes
    for i, (name, p) in enumerate(zip(class_names, probs)):
        y = 75 + i * 25
        cv2.rectangle(frame, (10, y - 15), (10 + int(p * 200), y), (255, 200, 0), -1)
        cv2.putText(frame, f"{name} {p * 100:.0f}%", (220, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("What is this? (Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
