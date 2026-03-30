from torch.utils.data import DataLoader
from demucs.iq_dataset import IQDataset


def build_iq_loader(path, batch_size, sources, segment):

    dataset = IQDataset(
        root=path,
        sources=sources,
        segment=segment
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    return loader