from fastapi import FastAPI, File, UploadFile
import librosa
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load TensorFlow model
model = tf.keras.models.load_model("mod.h5")

@app.post("/classify")
async def classify_audio(file: UploadFile = File(...)):
    # Load audio file
    y, sr = librosa.load(file.file, sr=None, mono=True)

    # Compute MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Normalize features
    mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)

    # Reshape features for TensorFlow model input
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    # Predict class probabilities using TensorFlow model
    preds = model.predict(mfcc)

    # Get predicted class label
    class_idx = np.argmax(preds)
    class_label = "Class " + str(class_idx)

    return {"class_label": class_label, "class_probs": preds.tolist()}
