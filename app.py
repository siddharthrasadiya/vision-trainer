"""
Vision Trainer - web app to collect photos, train and predict, all in the browser.

Run:
    venv\\Scripts\\activate
    python app.py
Then open http://127.0.0.1:5000
"""
import base64
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
MODEL_PATH = BASE / "model.keras"
LABELS_PATH = BASE / "labels.json"
IMG_SIZE = (224, 224)
MIN_IMAGES_PER_CLASS = 10
MAX_SAVE_WIDTH = 640  # shrink big photos to keep the dataset small

DATASET.mkdir(exist_ok=True)
app = Flask(__name__)

# Shared state
model_lock = threading.Lock()
model_state = {"model": None, "labels": []}
train_status = {"state": "idle", "epoch": 0, "epochs": 0, "message": "", "history": [],
                "val_accuracy": None}


# ---------- helpers ----------
def clean_name(name: str) -> str:
    name = (name or "").strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_-]", "", name)[:40]


def class_dir(name: str) -> Path:
    name = clean_name(name)
    if not name:
        abort(400, "Invalid name")
    return DATASET / name


def list_images(folder: Path):
    files = [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def decode_image(data_url: str):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(data_url), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR


def save_image(img, folder: Path) -> str:
    h, w = img.shape[:2]
    if w > MAX_SAVE_WIDTH:
        img = cv2.resize(img, (MAX_SAVE_WIDTH, int(h * MAX_SAVE_WIDTH / w)))
    fname = f"{folder.name}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.jpg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Could not encode image")
    (folder / fname).write_bytes(buf.tobytes())
    return fname


def get_model():
    with model_lock:
        if model_state["model"] is None and MODEL_PATH.exists() and LABELS_PATH.exists():
            from tensorflow import keras
            model_state["model"] = keras.models.load_model(MODEL_PATH)
            model_state["labels"] = json.loads(LABELS_PATH.read_text())
        return model_state["model"], model_state["labels"]


# ---------- pages ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dataset/<name>/<fname>")
def dataset_file(name, fname):
    return send_from_directory(class_dir(name), fname)


# ---------- classes ----------
@app.get("/api/classes")
def get_classes():
    out = []
    for d in sorted(p for p in DATASET.iterdir() if p.is_dir()):
        imgs = list_images(d)
        out.append({"name": d.name, "count": len(imgs),
                    "thumb": f"/dataset/{d.name}/{imgs[0].name}" if imgs else None})
    labels = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else []
    return jsonify({"classes": out, "model_labels": labels, "model_exists": MODEL_PATH.exists()})


@app.post("/api/classes")
def add_class():
    name = clean_name(request.json.get("name", ""))
    if not name:
        return jsonify(error="Use letters, numbers, - or _ only"), 400
    (DATASET / name).mkdir(exist_ok=True)
    return jsonify(name=name)


@app.delete("/api/classes/<name>")
def delete_class(name):
    d = class_dir(name)
    if d.exists():
        shutil.rmtree(d)
    return jsonify(ok=True)


# ---------- images ----------
@app.get("/api/classes/<name>/images")
def get_images(name):
    d = class_dir(name)
    if not d.exists():
        abort(404)
    return jsonify([{"file": p.name, "url": f"/dataset/{d.name}/{p.name}"} for p in list_images(d)])


@app.post("/api/classes/<name>/images")
def add_images(name):
    d = class_dir(name)
    d.mkdir(exist_ok=True)
    saved = []
    if request.is_json:  # photo from webcam (data URL)
        img = decode_image(request.json["image"])
        if img is not None:
            saved.append(save_image(img, d))
    for f in request.files.getlist("files"):  # uploaded files
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            saved.append(save_image(img, d))
    return jsonify(saved=saved, count=len(list_images(d)))


@app.delete("/api/classes/<name>/images/<fname>")
def delete_image(name, fname):
    if not re.fullmatch(r"[\w\-.]+", fname):
        abort(400)
    p = class_dir(name) / fname
    if p.exists():
        p.unlink()
    return jsonify(ok=True)


@app.delete("/api/classes/<name>/images")
def delete_all_images(name):
    d = class_dir(name)
    for p in list_images(d):
        p.unlink()
    return jsonify(ok=True)


# ---------- training ----------
def run_training(epochs: int):
    try:
        train_status.update(state="running", epoch=0, epochs=epochs, history=[],
                            val_accuracy=None, message="Loading TensorFlow...")
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        train_status["message"] = "Loading images..."
        train_ds, val_ds = keras.utils.image_dataset_from_directory(
            str(DATASET), validation_split=0.2, subset="both", seed=123,
            image_size=IMG_SIZE, batch_size=16)
        class_names = train_ds.class_names
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        augment = keras.Sequential([
            layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),
            layers.RandomZoom(0.1), layers.RandomContrast(0.2)])
        base = keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                              include_top=False, weights="imagenet")
        base.trainable = False

        inputs = keras.Input(shape=IMG_SIZE + (3,))
        x = augment(inputs)
        x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(class_names), activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        class Progress(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                train_status["message"] = f"Training epoch {epoch + 1} of {epochs}..."

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                row = {"epoch": epoch + 1,
                       "accuracy": round(float(logs.get("accuracy", 0)), 4),
                       "val_accuracy": round(float(logs.get("val_accuracy", 0)), 4),
                       "loss": round(float(logs.get("loss", 0)), 4)}
                train_status["history"].append(row)
                train_status.update(epoch=epoch + 1, val_accuracy=row["val_accuracy"])

        model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  callbacks=[Progress()], verbose=0)

        model.save(MODEL_PATH)
        LABELS_PATH.write_text(json.dumps(class_names))
        with model_lock:
            model_state["model"] = model
            model_state["labels"] = class_names
        acc = train_status["val_accuracy"] or 0
        train_status.update(state="done", message=f"Done! Accuracy on unseen photos: {acc * 100:.1f}%")
    except Exception as e:  # show the error in the web page
        train_status.update(state="error", message=f"{type(e).__name__}: {e}")


@app.post("/api/train")
def start_training():
    if train_status["state"] == "running":
        return jsonify(error="Training is already running"), 400
    classes = [d for d in DATASET.iterdir() if d.is_dir()]
    if len(classes) < 2:
        return jsonify(error="Add at least 2 items to train"), 400
    small = [d.name for d in classes if len(list_images(d)) < MIN_IMAGES_PER_CLASS]
    if small:
        return jsonify(error=f"Need at least {MIN_IMAGES_PER_CLASS} photos in: {', '.join(small)}"), 400
    epochs = max(1, min(50, int(request.json.get("epochs", 10))))
    threading.Thread(target=run_training, args=(epochs,), daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/train/status")
def training_status():
    return jsonify(train_status)


# ---------- prediction ----------
@app.post("/api/predict")
def predict():
    model, labels = get_model()
    if model is None:
        return jsonify(error="No trained model yet. Train first."), 400
    img = decode_image(request.json["image"])
    if img is None:
        return jsonify(error="Bad image"), 400
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, IMG_SIZE).astype("float32")[None, ...]
    with model_lock:
        probs = model(x, training=False).numpy()[0]
    results = sorted(({"label": l, "prob": float(p)} for l, p in zip(labels, probs)),
                     key=lambda r: r["prob"], reverse=True)
    return jsonify(results=results)


if __name__ == "__main__":
    # load the saved model in the background so the first prediction is fast
    threading.Thread(target=get_model, daemon=True).start()
    print("\n  Vision Trainer is running at http://127.0.0.1:5000")
    print("  Keep this window open while you use the app.\n")
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
