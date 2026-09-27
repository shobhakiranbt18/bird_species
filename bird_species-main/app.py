import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request
from PIL import Image
import json

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load class names
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

# Load model (no pretrained warning)
device = torch.device("cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("model/bird_model.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_filename = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image_filename = file.filename
            save_path = os.path.join(UPLOAD_FOLDER, image_filename)
            file.save(save_path)

            img = Image.open(save_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                predicted_class = output.argmax(1).item()
                prediction = class_names[predicted_class]

    return render_template(
        "index.html",
        prediction=prediction,
        image_filename=image_filename
    )

if __name__ == "__main__":
    app.run(debug=True)
