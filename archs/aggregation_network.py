import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from archs.detectron2.resnet import ResNet, BottleneckBlock

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

class AggregationNetworkObjectDetection(nn.Module):
    def __init__(
            self, 
            feature_dims, 
            device, 
            projection_dim=384, 
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timestep=[],
            num_timesteps=None,
            timestep_weight_sharing=False
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims    
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device
        self.save_timestep = save_timestep

        self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim // 4,
                    out_channels=projection_dim,
                    norm="GN",
                    num_norm_groups=num_norm_groups
                ),
                SelfAttention(projection_dim),  # Add self-attention mechanism
                # nn.LayerNorm(projection_dim)  # Add layer normalization
            )
            self.bottleneck_layers.append(bottleneck_layer)
            for t in save_timestep:
                # 1-index the layer name following prior work
                self.mixing_weights_names.append(f"timestep-{save_timestep}_layer-{l+1}")
        
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = torch.ones(len(self.bottleneck_layers) * len(save_timestep))
        self.mixing_weights = nn.Parameter(mixing_weights.to(device))

    def forward(self, batch):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        output_feature = None
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights)
        for i in range(len(mixing_weights)):
            # Share bottleneck layers across timesteps
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            if output_feature is None:
                output_feature = bottlenecked_feature
            else:
                output_feature += bottlenecked_feature
        return output_feature

class BaseModel(nn.Module):
    def __init__(self, device, dropout_prob=0.5):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(384, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_prob)
        self.to(device)  # Move modules to device

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)  # Apply dropout after pooling
        return x

class ObjectDetectionModel(nn.Module):
    def __init__(self, feature_dims, device, save_timestep, num_timesteps, dropout_prob=0.5, weight_decay=1e-4):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = AggregationNetworkObjectDetection(feature_dims, device, save_timestep=save_timestep, num_timesteps=num_timesteps)
        self.baseModel = BaseModel(device, dropout_prob=dropout_prob)
        self.regressor = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )
        self.to(device)  # Move modules to device

        # Apply weight decay to linear layers
        self.apply(self._init_weights)
        self.weight_decay = weight_decay

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        backbone_features = self.backbone(x)
        base_model_features = self.baseModel(backbone_features)
        bboxes = self.regressor(base_model_features.view(base_model_features.size(0), -1))
        return bboxes
