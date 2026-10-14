"""EfficientNet-based model for pneumonia classification."""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class PneumoniaClassifier(nn.Module):
    """
    Binary classifier built on EfficientNet-B0.
    Input: (B, 3, 224, 224)
    Output: (B, 1) logit — sigmoid for probability
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()

        # Load pretrained EfficientNet-B0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Get the input features of the final classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace the classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

        # Track the target layer for Grad-CAM (last conv layer)
        self.target_layer = self.backbone.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of PNEUMONIA class."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def get_model(pretrained: bool = True, device: str = "cpu") -> PneumoniaClassifier:
    """Factory function to create and move model to device."""
    model = PneumoniaClassifier(pretrained=pretrained)
    return model.to(device)


if __name__ == "__main__":
    # Quick smoke test
    model = get_model(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)
    print("Params:", sum(p.numel() for p in model.parameters()))