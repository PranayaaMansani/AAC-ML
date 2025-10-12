import os
from flask import Flask, request, jsonify, send_from_directory
from video1 import analyze_live_feed
from video2 import analyze_uploaded_video

# Get the absolute path to the directory where this script is located
basedir = os.path.abspath(os.path.dirname(__file__))
frontend_folder = os.path.join(basedir, '..', 'frontend')

# Create a Flask app instance
app = Flask(__name__)

@app.route('/')
def serve_html():
    """
    Serves the main HTML file from the frontend directory.
    """
    return send_from_directory(frontend_folder, 'video-analysis.html')

@app.route('/analyze-live', methods=['GET'])
def analyze_live():
    """
    Handles the request for live video emotion analysis.
    """
    return analyze_live_feed()

@app.route('/analyze-upload', methods=['POST'])
def analyze_upload():
    """
    Handles the uploaded video file.
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    return analyze_uploaded_video(file)

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, port=5000)
