import torch.nn as nn

class BoundingBoxPredictionNetwork(nn.Module):
    def __init__(self, input_channels):
        super(BoundingBoxPredictionNetwork, self).__init__()
        self.fc = nn.Linear(input_channels, 4)  # Output 4 values for each bounding box (x, y, w, h)

    def forward(self, x, num_boxes):
        # Pass through fully connected layer
        x = self.fc(x)
        # Reshape to separate bounding box predictions
        x = x.view(-1, num_boxes, 4)
        return x