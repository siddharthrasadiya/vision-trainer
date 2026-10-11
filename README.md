# Vision Trainer: train your own object detector in the browser

A web app that lets anyone teach a computer to recognise **any object** (pen, phone, watch, bag…) using their **laptop camera**: no coding, no dataset downloads, no GPU.

Add an item → take photos → click **Train** → point the camera and see live predictions.

![Collect screen](docs/collect.png)
![Predict screen](docs/predict.png)

## Features

- **Dynamic classes**: add or remove any item by name; the model adapts automatically
- **Webcam capture**: single shot or auto capture (~4 photos/sec), plus image upload
- **Dataset manager**: browse every training photo and delete bad ones
- **One-click training** with live progress and per-epoch accuracy
- **Real-time prediction** with confidence bars for every class and a "Not sure" threshold
- Runs fully **locally**: your photos never leave your computer

## How it works

1. **Transfer learning**: uses **MobileNetV2** pre-trained on ImageNet (1M+ images) as a frozen feature extractor, and trains only a small classification head on your photos. This gives high accuracy from ~50–100 photos per class in a few minutes on a CPU.
2. **Data augmentation** (flip, rotation, zoom, contrast) helps the model generalise beyond the exact photos taken.
3. A **Flask** backend handles storage, training (in a background thread) and inference; the frontend uses the browser **MediaDevices API** for the camera.

```
Browser (HTML/JS, webcam) ──► Flask REST API ──► TensorFlow / Keras (MobileNetV2)
                                   │
                                   └── dataset/<item>/*.jpg   (folder name = label)
```

## Tech stack

Python · TensorFlow / Keras · OpenCV · NumPy · Flask · HTML / CSS / JavaScript

## Getting started (Windows)

```bash
git clone https://github.com/<your-username>/vision-trainer.git
cd vision-trainer
setup.bat          # creates venv and installs dependencies (one time)
run_app.bat        # starts the app and opens http://127.0.0.1:5000
```

macOS / Linux:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Tip: add a **nothing** item with photos of your empty background, so the model can tell when no object is present.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/classes` | List items with photo counts |
| POST | `/api/classes` | Add an item |
| DELETE | `/api/classes/<name>` | Remove an item and its photos |
| GET / POST | `/api/classes/<name>/images` | List / add photos (webcam or upload) |
| DELETE | `/api/classes/<name>/images/<file>` | Delete a photo |
| POST | `/api/train` | Start training |
| GET | `/api/train/status` | Training progress |
| POST | `/api/predict` | Predict from an image |

## Command-line scripts

`collect.py`, `train.py` and `predict.py` do the same steps from the terminal with an OpenCV window.

## Roadmap

- [ ] Fine-tune the top layers of MobileNetV2 for higher accuracy
- [ ] Confusion matrix after training
- [ ] Export the model to TensorFlow.js / TFLite for browser and mobile
- [ ] Object detection with bounding boxes (YOLO)

## License

MIT
