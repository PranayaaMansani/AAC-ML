import os
import tempfile
from flask import jsonify
import cv2
from fer import FER
from collections import defaultdict
import moviepy.editor

# Emotion detector instance
detector = FER()

def analyze_uploaded_video(file):
    """
    Analyzes an uploaded video file for emotion distribution.
    """
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_dir = tempfile.mkdtemp()
    temp_filepath = os.path.join(temp_dir, file.filename)
    file.save(temp_filepath)
    print(f"File saved to {temp_filepath}")

    try:
        # Check if the video file can be opened
        cap = cv2.VideoCapture(temp_filepath)
        if not cap.isOpened():
            print(f"cv2.VideoCapture failed to open file: {temp_filepath}")
            return jsonify({"error": "Failed to open video file"}), 500

        emotion_counts = defaultdict(int)
        total_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use a sampling rate to avoid analyzing every frame
            if total_frames % 5 == 0:
                results = detector.detect_emotions(frame)
                if results:
                    emotions = results[0]["emotions"]
                    if emotions:
                        dominant = max(emotions, key=emotions.get)
                        emotion_counts[dominant] += 1
            
            total_frames += 1

        cap.release()
        
        total_emotions = sum(emotion_counts.values())
        emotion_percentages = {emotion: (count / total_emotions) * 100 for emotion, count in emotion_counts.items()} if total_emotions > 0 else {}
        
        print("Analysis complete:", emotion_percentages)
        return jsonify({"emotions": emotion_percentages})
        
    except Exception as e:
        print(f"Error during video analysis: {e}")
        return jsonify({"error": "Video analysis failed"}), 500
    finally:
        # Clean up temporary file and directory
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
