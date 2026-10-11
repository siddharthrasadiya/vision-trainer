"""
Step 1: Collect training images from your laptop camera.

Usage:
    python collect.py pen
    python collect.py mobile
    python collect.py hand
    python collect.py nothing      # empty background - important!

Controls (click the camera window first):
    SPACE  = save one photo
    C      = start/stop continuous capture (saves ~5 photos per second)
    Q      = quit
"""
import os
import sys
import time

import cv2

if len(sys.argv) < 2:
    print("Usage: python collect.py <label>   e.g. python collect.py pen")
    sys.exit(1)

label = sys.argv[1].strip().lower()
save_dir = os.path.join("dataset", label)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera. Close other apps using it (Zoom, Teams) and try again.")
    sys.exit(1)

count = len(os.listdir(save_dir))
continuous = False
last_save = 0.0

print(f"Collecting images for '{label}'. Already have {count}. Aim for 150+.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read from camera.")
        break

    now = time.time()
    if continuous and now - last_save > 0.2:  # ~5 photos per second
        cv2.imwrite(os.path.join(save_dir, f"{label}_{int(now * 1000)}.jpg"), frame)
        count += 1
        last_save = now

    display = frame.copy()
    status = "REC" if continuous else "paused"
    cv2.putText(display, f"{label}: {count} images [{status}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if continuous else (0, 255, 0), 2)
    cv2.putText(display, "SPACE=photo  C=continuous  Q=quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Collect images", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        cv2.imwrite(os.path.join(save_dir, f"{label}_{int(now * 1000)}.jpg"), frame)
        count += 1
    elif key == ord("c"):
        continuous = not continuous
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done. '{label}' now has {count} images in {save_dir}")
