import regex

import numpy as np
import torch

from bioner.model.CoNLLDataset import CoNLLDataset


def keep_only_printable_chars(input: str) -> str:
    return regex.sub('[^\\p{L}\\p{N}\\p{P}\\p{Sm}\\p{Sc}]', '', input).lower()


class NGramEncoder:
    def __init__(self, n: int):
        self.n = n
        self.words = []

    def create_encodings(self, dataset: CoNLLDataset):
        words = set()
        for document in dataset.documents:
            for sentence in document.sentences:
                for token in sentence.tokens:
                    n_grams = self.create_n_grams(token.text)
                    words.update(n_grams)
        self.words = list(words)

    def create_n_grams(self, token: str) -> [str]:
        word = '#' + keep_only_printable_chars(token) + '#'
        n_grams = []
        for i in range(len(word) - self.n):
            n_grams.append(word[i:i + self.n])
        return n_grams

    def get_embeddings_vector_size(self) -> int:
        if len(self.words) == 0:
            raise RuntimeError('Encoder not initialized, call create_encodings first')
        return len(self.words)

    def encode(self, phrase: str) -> [float]:
        """
        https://github.com/sebastianarnold/TeXoo/blob/514860d96decdf3ff6613dfcf0d27d9845ddcf60/texoo-core/src/main/java/de/datexis/encoder/impl/LetterNGramEncoder.java#L58-L69
        """
        vector = np.zeros(self.get_embeddings_vector_size())
        n_grams = self.create_n_grams(phrase)
        for n_gram in n_grams:
            try:
                index = self.words.index(n_gram)
            except ValueError:
                continue
            vector[index] = 1.0
        return vector


class TrigramEncoder(NGramEncoder):
    def __init__(self):
        super().__init__(n=3)
