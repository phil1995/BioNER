from torch.utils.data import DataLoader

from bioner.model.CoNLLDataset import CoNLLDataset
import random


class CoNLLDataLoader(DataLoader):
    """Custom DataLoader which supports shuffling the CoNLL dataset at the document level"""
    def __init__(self, dataset: CoNLLDataset, batch_size: int, shuffle: bool, num_workers: int, collate_fn=None):
        if shuffle:
            random.shuffle(dataset.documents)
        dataset.flatten_dataset()
        super().__init__(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
