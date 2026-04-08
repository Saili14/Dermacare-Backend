import torch
import json
from PIL import Image
import torchvision.transforms as transforms
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Correct path
MODEL_PATH = os.path.join(BASE_DIR, "models", "new_model", "model.pt")
LABELS_PATH = os.path.join(BASE_DIR, "models", "FR_model", "Exported_files", "labels.json")

# ✅ Load model ONCE
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# ✅ Load labels
with open(LABELS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# ✅ Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# MAIN FUNCTION (this is what API will call)
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    probs_list = probs.tolist()

    # 🔹 Top prediction
    max_idx = probs.argmax().item()
    top_label = CLASS_NAMES[max_idx]
    confidence = probs_list[max_idx]

    # 🔹 All probabilities
    prob_dict = {
        CLASS_NAMES[i]: float(probs_list[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "top": {
            "label": top_label,
            "confidence": confidence
        },
        "probabilities": prob_dict
    }