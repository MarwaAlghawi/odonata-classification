import requests as req
import os

def predict_image(img_path):
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    
    hf_token = os.environ.get('HF_TOKEN')
    
    # Try multiple models in order until one works
    models = [
        "google/vit-base-patch16-224",
        "microsoft/resnet-50",
        "facebook/convnext-tiny-224"
    ]
    
    for model_id in models:
        response = req.post(
            f"https://api-inference.huggingface.co/models/{model_id}",
            headers={"Authorization": f"Bearer {hf_token}"},
            data=img_bytes,
            timeout=15
        )
        
        if response.status_code == 200 and response.text.strip():
            try:
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
                    return {
                        "prediction": label,
                        "confidence": confidence,
                        "raw_label": results[0]['label'],
                        "model_used": model_id
                    }
            except:
                continue
    
    return {
        "prediction": "Model warming up",
        "confidence": 0.0,
        "message": "Please try again in 30 seconds"