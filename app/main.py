from flask import Flask, request, jsonify
from model import predict_image
from feedback import save_feedback
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "project": "Odonata Classification API",
        "version": "2.0",
        "endpoints": {
            "/predict": "POST — send image, get prediction",
            "/feedback": "POST — send correction feedback",
            "/health": "GET — check API status"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    # Save temporarily
    img_path = f"/tmp/{file.filename}"
    file.save(img_path)
    
    # Predict
    result = predict_image(img_path)
    
    # Clean up
    os.remove(img_path)
    
    return jsonify(result)

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    if not data or 'image_id' not in data or 'correct_label' not in data:
        return jsonify({"error": "Missing image_id or correct_label"}), 400
    
    save_feedback(data['image_id'], data['correct_label'], data.get('predicted_label'))
    return jsonify({"message": "Feedback saved! Thank you for helping improve the model."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)