import requests as req
import os

def predict_image(img_path):
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    
    hf_token = os.environ.get('HF_TOKEN')
    
    response = req.post(
        "https://api-inference.huggingface.co/models/google/vit-base-patch16-224",
        headers={"Authorization": f"Bearer {hf_token}"},
        data=img_bytes,
        timeout=30
    )
    
    # Handle errors
    if response.status_code != 200 or not response.text.strip():
        return {
            "prediction": "Model loading - try again in 20 seconds",
            "confidence": 0.0,
            "status_code": response.status_code,
            "model_version": "vit-base-patch16-224"
        }
    
    results = response.json()
    
    if isinstance(results, list) and len(results) > 0:
        top_label = results[0]['label'].lower()
        if 'dragonfly' in top_label:
            label = 'Dragonfly'
        elif 'damselfly' in top_label:
            label = 'Damselfly'
        else:
            label = results[0]['label']
        confidence = round(results[0]['score'] * 100, 1)
    else:
        label = str(results)
        confidence = 0.0
    
    return {
        "prediction": label,
        "confidence": confidence,
        "model_version": "vit-base-patch16-224"
    }