import fasttext

from bioner.model.CoNLLDataset import CoNLLDataset
from bioner.model.EncodedToken import EncodedToken
from bioner.model.encoder.Encoder import Encoder
from bioner.model.Token import Token


class FasttextEncoder(Encoder):
    def __init__(self, embeddings_file_path: str):
        self.model = fasttext.load_model(embeddings_file_path)

    def encode(self, dataset: CoNLLDataset):
        for document in dataset.documents:
            for sentence in document.sentences:
                encoded_tokens = [self.create_encoded_token_from_token(token) for token in sentence.tokens]
                sentence.tokens = encoded_tokens

    def get_embeddings_vector_size(self) -> int:
        return self.model.get_dimension()

    def create_encoded_token_from_token(self, token: Token):
        return EncodedToken(encoding=self.model[token.text], text=token.text,
                            start=token.start, end=token.end, tag=token.tag)
