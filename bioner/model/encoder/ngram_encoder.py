from collections import defaultdict

import regex

import numpy as np

from bioner.model.conll_dataset import CoNLLDataset


def keep_only_printable_chars(input: str) -> str:
    return regex.sub('[^\\p{L}\\p{N}\\p{P}\\p{Sm}\\p{Sc}]', '', input).lower()


class Vocabulary:
    MAX_CODE_LENGTH = 40

    def __init__(self):
        self.words = defaultdict(lambda: 0)

    def increment_word_count(self, word: str):
        self.words[word] += 1

    def truncate_vocabulary(self, min_word_frequency: int):
        truncated_words = defaultdict(lambda: 0)
        for key, value in self.words.items():
            if value >= min_word_frequency:
                truncated_words[key] = value
        self.words = truncated_words

    def update_huffman_codes(self):
        """Unnecessary to create huffman encoding! We only need the index of the sorted vocab"""
        # get vocabulary as sorted list
        vocab_words = self.convert_words_to_sorted_vocab_words()
        vocab_words.sort(key=lambda vocab_word: vocab_word.count)

        count = np.zeros(len(vocab_words) * 2 + 1)
        parent_node = np.zeros(len(vocab_words) * 2 + 1)
        binary = np.zeros(len(vocab_words) * 2 + 1)

        for i in range(len(vocab_words)):
            count[i] = vocab_words[i].count
        for i in range(len(vocab_words), len(vocab_words) * 2):
            count[i] = float('inf')

        pos1 = len(vocab_words) - 1
        pos2 = len(vocab_words)

        min1i = 0
        min2i = 0
        for i in range(len(vocab_words)):
            # First, find two smallest nodes 'min1, min2'
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1i = pos1
                    pos1 -= 1
                else:
                    min1i = pos2
                    pos2 += 1
            else:
                min1i = pos2
                pos2 += 1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2i = pos1
                    pos1 -= 1
                else:
                    min2i = pos2
                    pos2 += 1
            else:
                min2i = pos2
                pos2 += 1
            count[len(vocab_words) + i] = count[min1i] + count[min2i]
            parent_node[min1i] = len(vocab_words) + i
            parent_node[min2i] = len(vocab_words) + i
            binary[min2i] = 1

        # Now assign binary code to each vocabulary word
        code = np.zeros[Vocabulary.MAX_CODE_LENGTH]
        point = np.zeros[Vocabulary.MAX_CODE_LENGTH]
        b = 0
        i = 0
        for a in range(len(vocab_words)):
            b = a
            i = 0
            lcode = np.zeros[Vocabulary.MAX_CODE_LENGTH]
            lpoint = np.zeros[Vocabulary.MAX_CODE_LENGTH]
            while True:
                code[i] = binary[b]
                point[i] = b
                i += 1
                b = parent_node[b]
                if b == len(vocab_words * 2 - 2):
                    break

            lpoint[0] = len(vocab_words) - 2
            for idx in range(i):
                lcode[i - idx - 1] = code[idx]
                lpoint[i - idx] = point[idx] - len(vocab_words)

            vocab_words[a].huffman_idx = a

    def convert_words_to_sorted_vocab_words(self):
        vocab_words = []
        for word, count in self.words.items():
            vocab_words.append(VocabularyWord(word=word, count=count))
        vocab_words.sort(key=lambda vocab_word: vocab_word.count, reverse=True)
        return vocab_words


class LookupCache:
    def __init__(self, vocabulary):
        self.word_index_map = {}
        sorted_vocab_words = vocabulary.convert_words_to_sorted_vocab_words()
        for index, vocab_word in enumerate(sorted_vocab_words):
            self.word_index_map[vocab_word.word] = index

    def get_index_of_word(self, word):
        return self.word_index_map[word]

    def __len__(self):
        return len(self.word_index_map)


class VocabularyWord:
    def __init__(self, word, count):
        self.word = word
        self.count = count
        self.huffman_idx = None

    def __lt__(self, other):
        return self.count < other.count


class NGramEncoder:
    def __init__(self, n: int):
        self.n = n
        self.lookup_cache = None

    def create_encodings(self, dataset: CoNLLDataset, min_word_frequency: int = 10):
        vocabulary = Vocabulary()
        for document in dataset.documents:
            for sentence in document.sentences:
                for token in sentence.tokens:
                    n_grams = self.create_n_grams(token.text)
                    for n_gram in n_grams:
                        vocabulary.increment_word_count(n_gram)
        total = len(list(vocabulary.words.keys()))
        vocabulary.truncate_vocabulary(min_word_frequency=min_word_frequency)
        self.lookup_cache = LookupCache(vocabulary=vocabulary)
        print(f"trained {len(self.lookup_cache)} {self.n}-grams ({total} total)")

    def create_n_grams(self, token: str) -> [str]:
        word = '#' + keep_only_printable_chars(token) + '#'
        n_grams = []
        for i in range(len(word) - self.n + 1):
            n_grams.append(word[i:i + self.n])
        return n_grams

    def get_embeddings_vector_size(self) -> int:
        if self.lookup_cache is None:
            raise RuntimeError('Encoder not initialized, call create_encodings first')
        return len(self.lookup_cache)

    def encode(self, phrase: str) -> [float]:
        """
        https://github.com/sebastianarnold/TeXoo/blob/514860d96decdf3ff6613dfcf0d27d9845ddcf60/texoo-core/src/main/java/de/datexis/encoder/impl/LetterNGramEncoder.java#L58-L69
        """
        vector = np.zeros(self.get_embeddings_vector_size())
        n_grams = self.create_n_grams(phrase)
        for n_gram in n_grams:
            try:
                #index = self.words.index(n_gram)
                index = self.lookup_cache.get_index_of_word(n_gram)
            except KeyError:
                continue
            vector[index] = 1.0
        return vector

    def is_unknown(self, word) -> bool:
        n_grams = self.create_n_grams(word)
        for n_gram in n_grams:
            try:
                self.lookup_cache.get_index_of_word(n_gram)
            except KeyError:
                return True
        return False


class TrigramEncoder(NGramEncoder):
    def __init__(self):
        super().__init__(n=3)
