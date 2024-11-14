from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import os

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise tous les domaines ; pour des restrictions, remplacez par une liste d'origines spécifiques
    allow_credentials=True,
    allow_methods=["*"],  # Permet tous les méthodes HTTP
    allow_headers=["*"]   # Autorise tous les en-têtes
)

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize DenseNet model
model = models.densenet201(pretrained=False)  # Assuming you are using custom-trained weights
classifier = nn.Sequential(nn.Linear(1920, 256),
                           nn.ReLU(),
                           nn.Linear(256, 2),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier
model.load_state_dict(torch.load('coriander_vs_parsley_model_weights.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()

# Image transformation for the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the class names
class_names = ['Coriander', 'Parsley']

# Create a function to predict class
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(1, dim=1)
    return top_class.item()


@app.get("/")
def read_root():
    return {"message": "Welcome to Coriander vs Parsley API!"}

# Define the API endpoint for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()  # Read the image from the request
        image = Image.open(BytesIO(image_bytes))  # Open the image
    except Exception as e:
        return {"error": f"Invalid image file. Error: {str(e)}"}

    # Get the predicted class
    predicted_class = predict_image(image)
    predicted_label = class_names[predicted_class]  # Map the class index to the label
    return {"predicted_class": predicted_label}
