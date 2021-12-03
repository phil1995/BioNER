from abc import ABC, abstractmethod

from bioner.model.conll_dataset import CoNLLDataset


class Encoder(ABC):
    @abstractmethod
    def encode(self, dataset: CoNLLDataset):
        pass

    @abstractmethod
    def get_embeddings_vector_size(self) -> int:
        pass
