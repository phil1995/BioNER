from collections import defaultdict

import regex

import numpy as np


from bioner.model.CoNLLDataset import CoNLLDataset


def keep_only_printable_chars(input: str) -> str:
    return regex.sub('[^\\p{L}\\p{N}\\p{P}\\p{Sm}\\p{Sc}]', '', input).lower()


class Vocabulary:
    def __init__(self):
        self.words = defaultdict(lambda: 0)

    def update_word_count(self, word: str):
        self.words[word] += 1

    def truncate_vocabulary(self, min_word_frequency: int):
        truncated_words = defaultdict(lambda: 0)
        for key, value in self.words.items():
            if value >= min_word_frequency:
                truncated_words[key] = value
        self.words = truncated_words

    def update_huffman_codes(self):
        # get vocabulary as sorted list
        vocab_words = self.convert_words_to_vocab_words()
        vocab_words.sort(key=lambda vocab_word: vocab_word.count)
        print(vocab_words[0].count)

    def convert_words_to_vocab_words(self):
        vocab_words = []
        for word, count in self.words:
            vocab_words.append(VocabularyWord(word=word, count=count))
        return vocab_words

class VocabularyWord:
    def __init__(self, word, count):
        self.word = word
        self.count = count

    def __lt__(self, other):
        return self.count < other.count

class NGramEncoder:
    def __init__(self, n: int):
        self.n = n
        self.words = []
        self.vocabulary = None

    def create_encodings(self, dataset: CoNLLDataset, min_word_frequency: int = 10):
        self.vocabulary = Vocabulary()
        for document in dataset.documents:
            for sentence in document.sentences:
                for token in sentence.tokens:
                    n_grams = self.create_n_grams(token.text)
                    for n_gram in n_grams:
                        self.vocabulary.update_word_count(n_gram)
        total = len(list(self.vocabulary.words.keys()))
        self.vocabulary.truncate_vocabulary(min_word_frequency=min_word_frequency)
        self.words = list(self.vocabulary.words.keys())
        print(f"trained {len(self.words)}-{self.n}-grams ({total} total)")

    def create_n_grams(self, token: str) -> [str]:
        word = '#' + keep_only_printable_chars(token) + '#'
        n_grams = []
        for i in range(len(word) - self.n + 1):
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

    def is_unknown(self, word) -> bool:
        ngrams = self.create_n_grams(word)
        for ngram in ngrams:
            if self.vocabulary.words[ngram] == 0:
                return True
        return False


class TrigramEncoder(NGramEncoder):
    def __init__(self):
        super().__init__(n=3)
