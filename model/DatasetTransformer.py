import itertools
from fasttext.FastText import _FastText
from model.MedMentionsDataset import MedMentionsStructuredDataset


class DatasetTransformer:

    @staticmethod
    def transform_dataset(dataset: MedMentionsStructuredDataset):
        """
        Transforms a MedMentionsDataset object with the structure:
        |-Document:
          |-Sentence
            |-Token (text, tag)
        into: elements of
        [[text_j_1, ..., text_j_i], [tag_j_1, ..., tag_j_i]]
        (here for the j-th sentence)
        """
        flatten = itertools.chain.from_iterable
        rows = []
        for sentence in flatten(dataset.documents):
            texts = []
            tags = []
            for token in sentence.tokens:
                texts.append(token.text)
                tags.append(token.tag.value)
            rows.append([texts, tags])
        return rows


    @staticmethod
    def encode_dataset(dataset: MedMentionsStructuredDataset, encoder: _FastText):
        pass
