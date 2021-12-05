import os
import urllib

import fasttext
import requests
import tqdm

from bioner.model.conll_dataset import CoNLLDataset
from bioner.model.encoded_token import EncodedToken
from bioner.model.encoder.encoder import Encoder
from bioner.model.token import Token


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


class FastTextEmbedding:
    def __init__(self, embeddings_root: str, ngram_range: str, force_download: bool = False):
        self.filepath = os.path.join(embeddings_root, f"{ngram_range}-fastText-embeddings.bin")
        self.ngram_range = ngram_range
        self.download(force_download=force_download)

    def download(self, force_download: bool):
        if force_download or not os.path.isfile(self.filepath):
            url = FastTextEmbedding.get_url_for_ngram_range(ngram_range=self.ngram_range)
            self._urlretrieve(url)

    def _urlretrieve(self, url: str, chunk_size: int = 1024) -> None:
        # taken from:
        # https://github.com/pytorch/vision/blob/dcf5dc8747b94f70d1a3f1557df440fb567af95d/torchvision/datasets/utils.py#L30-L38
        with open(self.filepath, "wb") as file:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                total_file_size = int(response.headers['Content-Length'])
                num_bars = int(total_file_size / chunk_size)
                for chunk in tqdm.tqdm(
                        response.iter_content(chunk_size=chunk_size)
                        , total=num_bars
                        , unit='KB'
                        , desc=f"fastText {self.ngram_range}-grams"
                        , leave=True  # progressbar stays
                ):
                    if not chunk:
                        break
                    file.write(chunk)

    @staticmethod
    def get_url_for_ngram_range(ngram_range: str) -> str:
        if ngram_range == "3-4":
            # Fallback Address (slower): https://siasky.net/nACvdmBnYm86RAGGvsJLCaIDM2wCDoZK9Yy9_lLp9phgXA
            return "https://link.eu1.storjshare.io/jwsdq7ymfcnyxnqacofyckxjvyva/bioner%2Fpubmed.fasttext.3-4ngrams.neg5.1e-5_subs.bin?download"
        if ngram_range == "3-6":
            # Fallback Address (slower): https://siasky.net/nABUQPit8DTupo4eqidWdWIC9cozk14PiP8eIw2yYNB-BA
            return "https://link.eu1.storjshare.io/jxuer75wl52ijimisfsmwy46lpra/bioner%2Fpubmed.fasttext.3-6ngrams.neg5.1e-5_subs.bin?download"
        else:
            raise FastTextEmbeddingNotFoundException()


class FastTextEmbeddingNotFoundException(Exception):
    """
    Exception when the FastText embedding can't be downloaded because it does not exist
    """
