import torch

class VehicleAnnotationDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for vehicle annotations that returns features and targets.
    """
    def __init__(self, features, targets):
        """
        Args:
            features: tensor of shape [N, num_features]
            targets: tensor of shape [N, 1] or [N]
        """
        self.features = features
        # Keep targets as 2D [N, 1] for proper collation
        self.targets = targets if targets.dim() > 1 else targets.unsqueeze(1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]