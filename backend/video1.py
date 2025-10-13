from flask import jsonify
from collections import defaultdict

def analyze_live_feed():
    """
    Simulates live video analysis.
    """
    print("Simulating live analysis...")

    # Placeholder: Use a pre-trained face image for a consistent demo
    try:
        face_img = cv2.imread('sample_face.jpg')
        if face_img is None:
            face_img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.putText(face_img, "Placeholder", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    except:
        face_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(face_img, "Placeholder", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    results = detector.detect_emotions(face_img)
    
    # Convert results to frontend format
    emotion_counts = defaultdict(int)
    if results:
        emotions = results[0]["emotions"]
        for emotion, score in emotions.items():
            emotion_counts[emotion] = score * 100

    # Overall emotions list
    overall_emotions = [{"emotion": k, "percentage": round(v, 2)} for k, v in emotion_counts.items()]

    # Dominant emotion
    dominant = max(overall_emotions, key=lambda x: x["percentage"], default={"emotion": "None", "percentage": 0})

    return jsonify({
        "overall_emotions": overall_emotions,
        "detailed_analysis": {
            "dominant_emotion": dominant["emotion"].capitalize(),
            "confidence_score": dominant["percentage"]
        }
    })
