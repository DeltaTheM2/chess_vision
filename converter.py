import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class ReconstructedResNet(LightningModule):
    def __init__(self):
        super(ReconstructedResNet, self).__init__()

        # Adjusting the feature extractor to match the checkpoint's architecture
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),  # Adjust to match the checkpoint
            nn.BatchNorm2d(64),  # Adjust to 64 channels
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3, padding=1),  # Next conv layers
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        # Define classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 832)
        )


    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        # Classifier
        x = self.classifier(x)
        return x
# Load model weights
checkpoint = torch.load("C:\\Users\\smirz\\OneDrive\\Documents\\Coding Minds\\Chess vision\\chess-vision\\checkpoint.ckpt", map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']
model = ReconstructedResNet()
filtered_state_dict = {k: v for k, v in state_dict.items() if 'feature_extractor.7' not in k}

model.load_state_dict(checkpoint['state_dict'], strict=False)

# Save the model as .pt file
torch.save(model.state_dict(), 'C:\\Users\\smirz\\OneDrive\\Documents\\Coding Minds\\Chess vision\\chess-vision\\reconstructed_model.pt')
