import json
import os
from datetime import datetime

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), '..', 'logs', 'feedback.json')

def save_feedback(image_id, correct_label, predicted_label=None):
    # Load existing feedback
    feedback_data = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            try:
                feedback_data = json.load(f)
            except:
                feedback_data = []
    
    # Add new feedback entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "image_id": image_id,
        "predicted_label": predicted_label,
        "correct_label": correct_label,
        "was_correct": predicted_label == correct_label
    }
    
    feedback_data.append(entry)
    
    # Save back to file
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    
    print(f"✅ Feedback saved: {entry}")