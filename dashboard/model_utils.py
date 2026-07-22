import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class WaferClassifier:
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = [
            "Centre", "Donut", "Edge-Loc", "Edge-Ring", 
            "Loc", "Near-full", "Random", "Scratch"
        ]
        self.model = self._load_model(weights_path)
        
        # processing data to requirment of model 
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, weights_path):
        
        model = models.resnet18(weights=None)
        
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, len(self.classes))
        
        # loading model 
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image):
       
        img_np = np.array(image)
        
        if img_np.ndim == 3:
            img_np = img_np[:, :, 0]
            
        # making the model understand the colours in image

        high_contrast_matrix = np.zeros_like(img_np, dtype=np.uint8)
        high_contrast_matrix[img_np == 1] = 127
        high_contrast_matrix[img_np >= 2] = 255
            
       
        three_channel_matrix = np.stack([high_contrast_matrix, high_contrast_matrix, high_contrast_matrix], axis=-1)
        
       
        final_image = Image.fromarray(three_channel_matrix)
            
       
        img_tensor = self.transform(final_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            # sigmoid function for multi label classification
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
         # creating the result 
        results = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        return results