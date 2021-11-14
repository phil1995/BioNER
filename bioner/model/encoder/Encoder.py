from abc import ABC, abstractmethod

from bioner.model.CoNLLDataset import CoNLLDataset


class Encoder(ABC):
    @abstractmethod
    def encode(self, dataset: CoNLLDataset):
        pass

    @abstractmethod
    def get_embeddings_vector_size(self) -> int:
        pass
