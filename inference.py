import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# --- Extracted Champion Architecture ---
class ChampionMLP(nn.Module):
    def __init__(self, in_dim=784, out_dim=15, dropout=0.2):
        super().__init__()
        # Funnel Architecture: Start wide, get narrow
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Layer 4
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Output Layer
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------

def predict_image(image_path, model_path="champion_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    model = ChampionMLP(in_dim=784, out_dim=15, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    
    # Load and transform image
    img = Image.open(image_path).convert('L')
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        
    return predicted.item()

if __name__ == "__main__":
    print("Inference script loaded. Run predict_image('path_to_image.png') to test.")
