import torch
import torchvision.models as models
import torch.nn as nn

# same structure as training
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

# load trained weights
model.load_state_dict(torch.load("models/new_model/model.pth", map_location="cpu"))

model.eval()

# dummy input
example = torch.randn(1, 3, 224, 224)

# convert
traced_model = torch.jit.trace(model, example)

# save
torch.jit.save(traced_model, "models/new_model/model.pt")

print("✅ TorchScript model saved!")