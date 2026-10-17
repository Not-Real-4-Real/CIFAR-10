import torch
from models.ViT_homemade import ViT
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_models" / "ViT_best.pth"

# 1. Select the device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 2. Reconstruct the original architecture
model = ViT(
    num_classes=10,
    patch_size=4,
    d_model=192,
    num_heads=3,
    d_ff=384,
    num_layers=6,
    dropout=0.0
)

# 3. Load the learned parameters
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

# 4. Move the model to the device
model = model.to(device)

# 5. Prepare it for inference
model.eval()

print("Model loaded successfully!")