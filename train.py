"""
Step 2: Train the model on your images.

Uses "transfer learning": MobileNetV2 already knows how to see shapes and
edges (trained on millions of images). We only teach it the last step:
your own classes (pen, mobile, hand, nothing). This trains in a few
minutes on a normal laptop, no GPU needed.

Usage:
    python train.py
"""
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

# Load images from dataset/<label>/*.jpg  (folder name = label)
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
class_names = train_ds.class_names
print("Classes found:", class_names)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Random changes so the model doesn't just memorise your photos
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
])

# Pre-trained base model (frozen)
base = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # scale pixels to [-1, 1] like MobileNetV2 expects
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

loss, acc = model.evaluate(val_ds)
print(f"\nValidation accuracy: {acc * 100:.1f}%")

model.save("model.keras")
with open("labels.json", "w") as f:
    json.dump(class_names, f)
print("Saved model.keras and labels.json")
