import numpy as np
import torch

from bioner.model.conll_dataset import CoNLLDataset


class PositionEncoder:
    def get_embeddings_vector_size(self) -> int:
        return 4

    @staticmethod
    def encode_dataset(dataset: CoNLLDataset):
        for document in dataset.documents:
            begin_doc = True
            for sentence_idx, sentence in enumerate(document.sentences):
                end_doc = sentence_idx == len(document.sentences) - 1
                begin_sentence = True
                for token_idx, token in enumerate(sentence.tokens):
                    end_sentence = token_idx == len(sentence.tokens) - 1
                    position_encoding = PositionEncoder.create_vector(begin_doc=begin_doc and begin_sentence,
                                                                      begin_sentence=begin_sentence,
                                                                      end_sentence=end_sentence,
                                                                      end_document=end_doc and end_sentence)
                    token.encoding = np.concatenate((position_encoding, token.encoding))
                    begin_sentence = False
                begin_doc = False

    @staticmethod
    def create_vector(begin_doc: bool, begin_sentence: bool, end_sentence: bool, end_document: bool) -> [float]:
        features = [begin_doc, begin_sentence, end_sentence, end_document]
        vector = np.zeros(len(features))
        for i, feature in enumerate(features):
            vector[i] = 1 if feature else 0
        return vector
