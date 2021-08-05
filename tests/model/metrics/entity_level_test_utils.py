import numpy as np

from bioner.model.BIO2Tag import BIO2Tag


def transform_tag_to_index(tag: BIO2Tag) -> int:
    return BIO2Tag.get_index(tag)


def transform_tag_to_prob(tag: BIO2Tag) -> [float]:
    prob = np.random.rand(3)
    # swap max to tag index
    tag_index = BIO2Tag.get_index(tag)
    argmax = np.argmax(prob)
    if tag_index != argmax:
        prob[[tag_index, argmax]] = prob[[argmax, tag_index]]
    return prob


def transform_index_to_prob(index: int) -> [float]:
    prob = np.random.rand(3)
    # swap max to index
    argmax = np.argmax(prob)
    if index != argmax:
        prob[[index, argmax]] = prob[[argmax, index]]
    return prob