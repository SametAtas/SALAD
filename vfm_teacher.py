import torch
import torch.nn as nn
import torch.nn.functional as F

class VFMTeacher(nn.Module):
    def __init__(self, out_channels=384):
        super().__init__()
        # Load DINOv2 small, which naturally outputs 384-dimensional features
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Freeze the backbone completely
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.out_channels = out_channels
        
    def forward(self, x):
        # x is (B, 3, 256, 256)
        B, C, H, W = x.shape
        
        # DINOv2 expects image sizes to be multiples of 14. 
        # 256 is not a multiple of 14 (256/14 = 18.28).
        # We resize to 252x252 (18x18 patches) for the ViT forward pass.
        x_resized = F.interpolate(x, size=(252, 252), mode='bilinear', align_corners=False)
        
        # Extract features from DINOv2
        # forward_features returns a dict containing 'x_norm_patchtokens'
        with torch.no_grad():
            features = self.backbone.forward_features(x_resized)['x_norm_patchtokens']
            
        # features shape: (B, 18*18, 384)
        # Reshape to spatial dimensions (B, 384, 18, 18)
        features = features.reshape(B, 18, 18, self.out_channels).permute(0, 3, 1, 2)
        
        # SALAD's student (PDN Medium) outputs spatial dimensions of 64x64 for a 256x256 image.
        # To compute the pixel-wise Student-Teacher loss, the spatial dimensions must match.
        # We upsample the 18x18 VFM features to 64x64 to match the student.
        features_upsampled = F.interpolate(features, size=(64, 64), mode='bilinear', align_corners=False)
        
        return features_upsampled
