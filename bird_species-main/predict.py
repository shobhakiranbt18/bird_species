import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import json

# ----------------------------------------
# LOAD MODEL + CLASS NAMES (200 classes)
# ----------------------------------------
model_path = r"C:\Bird_Species\model\bird_model.pth"
classes_file = r"C:\Bird_Species\model\class_names.json"

with open(classes_file, "r") as f:
    class_names = json.load(f)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 200)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ----------------------------------------
# IMAGE TRANSFORM
# ----------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------------------
# PREDICT + SHOW IMAGE (TRANSPARENT BOX)
# ----------------------------------------
def predict_and_show(img_path):

    # Load input image
    img = Image.open(img_path).convert("RGB")

    # Preprocess
    input_tensor = transform(img).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    species_name = class_names[predicted.item()]

    # Convert to RGBA to add transparency
    img_rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = img.size

    # Font
    try:
        font = ImageFont.truetype("arial.ttf", 35)
    except:
        font = ImageFont.load_default()

    text = species_name

    # Text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = 20
    box_x1 = (width - text_w) // 2 - padding
    box_y1 = height - text_h - 50
    box_x2 = box_x1 + text_w + padding * 2
    box_y2 = box_y1 + text_h + padding

    # Transparent box
    draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill=(0, 0, 0, 150))

    # Add text
    text_x = (width - text_w) // 2
    text_y = box_y1 + (padding // 2)
    draw.text((text_x, text_y), text, font=font, fill="white")

    final_img = Image.alpha_composite(img_rgba, overlay)

    # -------------------------------
    # FIRST SHOW IMAGE
    # -------------------------------
    final_img.show()

    # -------------------------------
    # THEN PRINT SPECIES NAME AFTER
    # -------------------------------
    print("\nPredicted Bird Species:", species_name)


# ----------------------------------------
# GIVE IMAGE PATH HERE
# ----------------------------------------
image_path = r"C:\Bird_Species\Predict\class3.jpg"
predict_and_show(image_path)
