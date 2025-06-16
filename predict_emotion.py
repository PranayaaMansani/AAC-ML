from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
import librosa
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load model and feature extractor
model_name = "superb/wav2vec2-base-superb-er"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Fix file path (use raw string)
file_path = r"C:\Users\kanda\Desktop\AAC\Actor_03\03-01-08-02-01-01-03.wav"

# Load and preprocess audio
audio_input, _ = librosa.load(file_path, sr=16000)
inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt")

# Predict emotion
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_id = torch.argmax(logits).item()

# Print prediction
emotion_labels = model.config.id2label
print("Predicted Emotion:", emotion_labels[predicted_class_id])
