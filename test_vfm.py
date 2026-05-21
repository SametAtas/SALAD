import torch
print("Testing AnomalyVFM loading via torch.hub...")
try:
    model = torch.hub.load("MaticFuc/AnomalyVFM", "anomalyvfm_dinov2", trust_remote_code=True)
    print("SUCCESS: Loaded DINOv2 AnomalyVFM backbone!")
except Exception as e:
    print(f"FAILED: {e}")
