import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import time
import argparse

#config (optimized for a 32gb m1 max macbook pro)
TRAIN_PATH = 'dataset_70_TRAIN.npz'
VAL_PATH = 'dataset_15_VAL.npz'

BATCH_SIZE = 256         
NUM_WORKERS = 8          
EPOCHS = 20
LEARNING_RATE = 1e-4     

NUM_CLASSES = 8
MODEL_CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]

class ShallowVanillaCNN(nn.Module):
    def __init__(self, num_classes):
        super(ShallowVanillaCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_architecture(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == "vanilla_cnn":
        return ShallowVanillaCNN(num_classes)
    else:
        raise ValueError(f"Unknown architecture option: {model_name}")

# focal loss setup
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_factor * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()

def get_loss_criterion(loss_name):
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "focal":
        return MultiLabelFocalLoss(alpha=0.25, gamma=2.0)
    else:
        raise ValueError(f"Unknown loss configuration: {loss_name}")

# data loading and preprocessing
class FastWaferDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.x_data = data['x']
        self.y_data = data['y']
        self.transform = transform

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx):
        img_array = self.x_data[idx]
        labels = self.y_data[idx]
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
        if self.transform: 
            img_tensor = self.transform(img_tensor)
        return img_tensor, torch.from_numpy(labels).float()

def run_experiment(model_type, loss_type):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    output_weights_name = f"weights_{model_type}_{loss_type}.pth"
    
    print(f"\nStarting training: Model={model_type}, Loss={loss_type}")
    print(f"Output weights: {output_weights_name}")

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = FastWaferDataset(TRAIN_PATH, transform=train_transforms)
    val_dataset = FastWaferDataset(VAL_PATH, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = get_architecture(model_type, NUM_CLASSES).to(device)
    criterion = get_loss_criterion(loss_type)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(EPOCHS):
        # training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        
        # val phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        
        # checkpointing
        status = ""
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), output_weights_name)
            status = " (saved best)"

        print(f"Epoch {epoch+1:02d}/{EPOCHS} - Train Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}{status}")

    

if __name__ == '__main__':
    # choosing which model/loss type to run 
    parser = argparse.ArgumentParser(description="Wafer Defect Training Pipeline")
    parser.add_argument('--model', type=str, choices=['resnet18', 'vanilla_cnn', 'all'], default='all')
    parser.add_argument('--loss', type=str, choices=['bce', 'focal', 'all'], default='all')
    args = parser.parse_args()

    models_to_run = ['resnet18', 'vanilla_cnn'] if args.model == 'all' else [args.model]
    losses_to_run = ['bce', 'focal'] if args.loss == 'all' else [args.loss]

    for m in models_to_run:
        for l in losses_to_run:
            run_experiment(m, l)