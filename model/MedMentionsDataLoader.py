import torch.utils.data.dataloader

from model.MedMentionsDataset import MedMentionsDataset
import random


class MedMentionsDataLoader(torch.DataLoader):
    """Custom DataLoader which supports shuffling the MedMentions dataset at the document level"""
    def __init__(self, dataset: MedMentionsDataset, batch_size: int, shuffle: bool, num_workers: int):
        if shuffle:
            random.shuffle(dataset.documents)
        super().__init__(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
