import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import requests
from io import BytesIO
from flask import Flask, render_template, request, jsonify
import base64
import uuid

# Import functions from our main script
from celeb_recognition import predict_celebrity, preprocess_image

app = Flask(__name__)

# Global variables
MODEL = None
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global MODEL
    if MODEL is None:
        MODEL_PATH = 'celebrity_recognition_model.h5'
        if os.path.exists(MODEL_PATH):
            print("Loading model...")
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print("Model not found. Please train the model first using celeb_recognition.py")
    return MODEL

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    model = load_model()
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    # Check if the request has the URL
    if 'url' in request.form and request.form['url']:
        image_url = request.form['url']
        try:
            result = predict_celebrity(model, image_url)
            if result:
                return jsonify(result)
            else:
                return jsonify({'error': 'Failed to process the image from URL'})
        except Exception as e:
            return jsonify({'error': f'Error processing URL: {str(e)}'})
    
    # Check if the request has the file part
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            # Save the file with a unique filename
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            try:
                # Process the image and make prediction
                result = predict_celebrity(model, filepath)
                if result:
                    # Add the image path to the result
                    result['image_path'] = filepath
                    return jsonify(result)
                else:
                    return jsonify({'error': 'Failed to process the uploaded image'})
            except Exception as e:
                return jsonify({'error': f'Error processing upload: {str(e)}'})
        else:
            return jsonify({'error': f'File type not allowed. Please upload {", ".join(ALLOWED_EXTENSIONS)}'})
    
    # If neither URL nor file is provided
    else:
        return jsonify({'error': 'No URL or file provided'})

if __name__ == '__main__':
    # Try to load the model at startup
    load_model()
    app.run(debug=True, port=5000) 