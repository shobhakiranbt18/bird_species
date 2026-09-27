import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import sys
import json

# ------------------------------------------
# PATHS (update if necessary)
# ------------------------------------------
model_path = r"C:\Bird_Species\model\bird_model.pth"
# Optionally, store class names in a json file during training
class_names_file = r"C:\Bird_Species\model\class_names.json"

# ------------------------------------------
# CHECK USER INPUT
# ------------------------------------------
if len(sys.argv) < 2:
    print("\nERROR: Please give a folder or image path as input.")
    print("Example:")
    print("python bird_classification.py C:\\Bird_Species\\Predict\\bird1.jpg\n")
    sys.exit()

input_path = sys.argv[1]

if not os.path.exists(input_path):
    print(f"\nERROR: Path not found: {input_path}\n")
    sys.exit()

# ------------------------------------------
# IMAGE TRANSFORM
# ------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------------------------
# LOAD CLASS LABELS
# ------------------------------------------
# Option 1: Load from JSON (recommended)
if os.path.exists(class_names_file):
    with open(class_names_file, "r") as f:
        classes = json.load(f)
else:
    # Option 2: Hardcode class names if JSON not available
    classes = ["Class1", "Class2"]  # replace with your real class names

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_classes = len(classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ------------------------------------------
# FUNCTION: Predict one image
# ------------------------------------------
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1)[0]
        confidence, pred_class = torch.max(probs, 0)

    return classes[pred_class], confidence.item() * 100

# ------------------------------------------
# RUN PREDICTION
# ------------------------------------------
print("\n===================== RESULTS =====================")

if os.path.isfile(input_path):
    # single image
    species, conf = predict(input_path)
    print(f"\nImage            : {os.path.basename(input_path)}")
    print(f"Predicted Species: {species}")
    print(f"Confidence       : {conf:.2f}%")
    print("--------------------------------------------------")
else:
    # folder of images
    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.webp')):
            img_path = os.path.join(input_path, filename)
            species, conf = predict(img_path)
            print(f"\nImage            : {filename}")
            print(f"Predicted Species: {species}")
            print(f"Confidence       : {conf:.2f}%")
            print("--------------------------------------------------")

print("\n===================== DONE =====================\n")

