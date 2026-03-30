import torch
from torch.utils.data import Dataset
import numpy as np
import os

class IQDataset(Dataset):
    def __init__(self, path, segment=262144, sources=2):
        """
        path: Folder containing subfolders for each source
        segment: Number of I/Q samples per training chunk
        """
        self.path = path
        self.segment = segment
        self.files = [f for f in os.listdir(path) if f.endswith('.npy') or f.endswith('.dat')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the mixture (input)
        # This assumes your data is stored as [2, length] float32
        data = np.load(os.path.join(self.path, self.files[idx]))
        
        # Random crop to 'segment' length
        if data.shape[-1] > self.segment:
            start = np.random.randint(0, data.shape[-1] - self.segment)
            data = data[:, start:start + self.segment]
        
        # Convert to complex tensor
        # Demucs loss functions often prefer complex input, 
        # but the model internalizes it as 2-channel real.
        mix = torch.from_numpy(data[0] + 1j * data[1]) 
        sources = torch.stack([mix * 0.5, mix * 0.5]) # Placeholder: Replace with actual ground truth sources!
        
        return mix, sources