import torch
from torchvision import transforms
from PIL import Image

# ----- CONFIG -----
MODEL_PATH = "bird_model.pth"   # your trained model
IMAGE_PATH = "test.jpg"         # image to test
CLASS_NAMES = ["sparrow", "eagle", "pigeon"]  # same order as training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- LOAD MODEL -----
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# ----- IMAGE TRANSFORM (same as training) -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----- LOAD IMAGE -----
image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# ----- PREDICTION -----
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print("Predicted Bird Species:", CLASS_NAMES[predicted.item()])

