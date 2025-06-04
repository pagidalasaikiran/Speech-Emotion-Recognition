import gradio as gr
from extract_features import extract_features
import pickle
import numpy as np

def emotion_from_file(file):
    with open("model.pkl", "rb") as f:
        clf = pickle.load(f)
    feat = extract_features(file.name).reshape(1, -1)
    return clf.predict(feat)[0]

gr.Interface(fn=emotion_from_file, inputs="file", outputs="text", title="Speech Emotion Recognizer").launch()
