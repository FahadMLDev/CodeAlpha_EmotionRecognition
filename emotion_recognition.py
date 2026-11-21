"""
CodeAlpha Task 2: Emotion Recognition from Speech
Dataset: RAVDESS (speech only) -> folder "Audio_Speech_Actors_01-24" containing Actor_01 ... Actor_24
Pipeline: audio -> MFCC features -> 1D CNN -> emotion class
"""
import os, glob
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------- Settings ----------
DATA_DIR = "Audio_Speech_Actors_01-24"   # change if your folder has another name
SR = 22050            # sample rate
DURATION = 3.0        # seconds used from each clip
N_MFCC = 40           # number of MFCC coefficients
MAX_FRAMES = 130      # fixed time length (3 s at SR=22050, hop=512)
TEST_ACTORS = {19, 20, 21, 22, 23, 24}   # speaker-independent test set
EPOCHS = 60
EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}

# ---------- Step 1: Extract MFCC features ----------
def extract_mfcc(path):
    y, _ = librosa.load(path, sr=SR, duration=DURATION, offset=0.5)
    y, _ = librosa.effects.trim(y, top_db=25)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)      # (40, frames)
    mfcc = mfcc[:, :MAX_FRAMES]
    if mfcc.shape[1] < MAX_FRAMES:                                # pad short clips
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])))
    return mfcc.T                                                 # (frames, 40)

CACHE = "features.npz"
if os.path.exists(CACHE):
    d = np.load(CACHE, allow_pickle=True)
    X, labels, actors = d["X"], d["labels"], d["actors"]
    print("Loaded cached features:", X.shape)
else:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True))
    print("Audio files found:", len(files))
    assert files, f"No .wav files found in '{DATA_DIR}'. Check the folder name/path."
    X, labels, actors = [], [], []
    for i, f in enumerate(files, 1):
        parts = os.path.basename(f)[:-4].split("-")
        labels.append(EMOTIONS[parts[2]])
        actors.append(int(parts[6]))
        X.append(extract_mfcc(f))
        if i % 200 == 0:
            print(f"  processed {i}/{len(files)}")
    X, labels, actors = np.array(X), np.array(labels), np.array(actors)
    np.savez(CACHE, X=X, labels=labels, actors=actors)

# ---------- Step 2: Normalize, encode labels, split by speaker ----------
le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)
test_mask = np.isin(actors, list(TEST_ACTORS))
X_train, y_train = X[~test_mask], y[~test_mask]
X_test, y_test = X[test_mask], y[test_mask]
mean, std = X_train.mean(axis=(0, 1)), X_train.std(axis=(0, 1)) + 1e-8
X_train, X_test = (X_train - mean) / std, (X_test - mean) / std
print("Train:", X_train.shape, "Test:", X_test.shape, "Classes:", classes)

# ---------- Step 3: Build the 1D CNN ----------
def conv_block(x, filters):
    x = layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    return layers.Dropout(0.3)(x)

inp = layers.Input(shape=X_train.shape[1:])
x = conv_block(inp, 64)
x = conv_block(x, 128)
x = conv_block(x, 256)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(len(classes), activation="softmax")(x)
model = models.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------- Step 4: Train ----------
cbs = [callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor="val_accuracy"),
       callbacks.ReduceLROnPlateau(factor=0.5, patience=5)]
hist = model.fit(X_train, y_train, validation_split=0.15, epochs=EPOCHS,
                 batch_size=32, callbacks=cbs, verbose=2)

# ---------- Step 5: Evaluate ----------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy (unseen speakers): {acc:.3f}")
pred = model.predict(X_test, verbose=0).argmax(axis=1)
print(classification_report(y_test, pred, target_names=classes, zero_division=0))

# ---------- Step 6: Plots + save model ----------
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(hist.history["accuracy"], label="train")
plt.plot(hist.history["val_accuracy"], label="val"); plt.title("Accuracy"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(hist.history["loss"], label="train")
plt.plot(hist.history["val_loss"], label="val"); plt.title("Loss"); plt.legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150)

model.save("emotion_model.keras")
print("Saved: confusion_matrix.png, training_curves.png, emotion_model.keras")