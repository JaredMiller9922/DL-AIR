import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

def complex_to_channels(x):

    real = x.real
    imag = x.imag

    return torch.stack([real, imag], dim=0)


class IQDataset(Dataset):

    """
    RF dataset loader for Demucs.

    Each directory must contain:

        mixture.iq
        source1.iq
        source2.iq
        ...

    All files are complex64 raw IQ samples.
    """

    def __init__(self, root, sources=2, segment=None):

        self.root = Path(root)
        self.segment = segment
        self.sources = sources

        self.samples = sorted([x for x in self.root.iterdir() if x.is_dir()])


    def load_iq(self, path):

        data = np.fromfile(path, dtype=np.complex64)

        return torch.from_numpy(data)


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        sample_dir = self.samples[idx]

        mixture = self.load_iq(sample_dir / "mixture.iq")

        src_list = []

        for i in range(self.sources):
            src = self.load_iq(sample_dir / f"source{i+1}.iq")
            src_list.append(src)

        sources = torch.stack(src_list)

        if self.segment and mixture.shape[-1] > self.segment:

            start = torch.randint(
                0,
                mixture.shape[-1] - self.segment,
                (1,)
            ).item()

            mixture = mixture[start:start+self.segment]
            sources = sources[:, start:start+self.segment]

        # convert complex → channels

        mix = complex_to_channels(mixture)

        sources = torch.stack([
            complex_to_channels(s)
            for s in sources
        ])

        return mix, sources