from extract_features import extract_features
import numpy as np
import pickle

def predict_emotion(file_path, model_path="model.pkl"):
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    feat = extract_features(file_path).reshape(1, -1)
    return clf.predict(feat)[0]
