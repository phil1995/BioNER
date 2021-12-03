import numpy as np

from bioner.model.conll_dataset import CoNLLDataset
from bioner.model.encoded_token import EncodedToken
from bioner.model.encoder.encoder import Encoder
from bioner.model.encoder.ngram_encoder import TrigramEncoder
from bioner.model.encoder.position_encoder import PositionEncoder
from bioner.model.encoder.surface_encoder import SurfaceEncoder
from bioner.model.token import Token


class DATEXISEncoder(Encoder):

    def __init__(self):
        self.trigram_encoder = TrigramEncoder()
        self.surface_encoder = SurfaceEncoder()
        self.position_encoder = PositionEncoder()

    def encode(self, dataset: CoNLLDataset):
        for document in dataset.documents:
            for sentence in document.sentences:
                encoded_tokens = [self.create_encoded_token_from_token(token) for token in sentence.tokens]
                sentence.tokens = encoded_tokens
        self.position_encoder.encode_dataset(dataset)

    def create_encoded_token_from_token(self, token: Token):
        trigram_encoding = self.trigram_encoder.encode(token.text)
        surface_encoding = self.surface_encoder.encode(token.text)

        return EncodedToken(encoding=np.concatenate((surface_encoding, trigram_encoding)), text=token.text,
                            start=token.start, end=token.end, tag=token.tag)

    def learn_trigram_encoding(self, dataset: CoNLLDataset):
        self.trigram_encoder.create_encodings(dataset)

    def get_embeddings_vector_size(self) -> int:
        return self.trigram_encoder.get_embeddings_vector_size() + self.position_encoder.get_embeddings_vector_size() \
               + self.surface_encoder.get_embedding_vector_size()
