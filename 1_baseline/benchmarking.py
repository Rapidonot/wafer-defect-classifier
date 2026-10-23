import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 12 

# config (optmised for a 32gb m1 max macbook pro)
TEST_PATH = 'dataset_15_TEST.npz'
MODEL_PATH = 'research_resnet18_bce.pth'
BATCH_SIZE = 256
NUM_CLASSES = 8
MODEL_CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]

class TestWaferDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.x_data = data['x']
        self.y_data = data['y']
        self.transform = transform

    def __len__(self): return len(self.x_data)

    def __getitem__(self, idx):
        img_array = self.x_data[idx]
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
        if self.transform: img_tensor = self.transform(img_tensor)
        return img_tensor, torch.from_numpy(self.y_data[idx]).float()

def evaluate_and_plot():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model running on: {device}")

    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
   
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print(f"load error: {e}")
        return
        
    model = model.to(device).eval()

   #setup datafor testing 
    test_transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = TestWaferDataset(TEST_PATH, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []


    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # using sigmoid function
            preds = (torch.sigmoid(outputs) > 0.50).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

   
    print("\n" + "="*60)
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=MODEL_CLASSES, zero_division=0))
    print("="*60)

    #confusion martrix
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for i, (matrix, name) in enumerate(zip(mcm, MODEL_CLASSES)):
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False, annot_kws={"size": 10})
        axes[i].set_title(f"Class: {name}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Predicted", fontsize=10)
        axes[i].set_ylabel("Actual", fontsize=10)

    plt.tight_layout()
    plt.savefig('confusion_matrix.tif', format='tif', dpi=500, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    evaluate_and_plot()