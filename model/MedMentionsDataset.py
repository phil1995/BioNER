import csv
import itertools as it

from fasttext.FastText import _FastText
from torch.utils.data import Dataset

from model.BIO2Tag import BIO2Tag
from model.Document import Document
from model.EncodedToken import EncodedToken
from model.Sentence import Sentence
from model.Token import Token


class MedMentionsDataset(Dataset):
    """
    The MedMentionsDataset with embeddings

    Note: Before using it as a PyTorch.Dataset you should call flatten_dataset()
    This is because the dataset is structured in the following way:
    |-Document:
      |-Sentence
        |-Token (text, tag, embedding)

    to make it possible to shuffle at the document level.
    """
    DOC_START = "-DOCSTART-"

    def __init__(self, data_file_path: str, encoder: _FastText):
        self.documents = self.read_documents(data_file_path, encoder)
        self.sentences = None

    def read_documents(self, data_file_path: str, encoder: _FastText):
        documents = []
        with open(data_file_path, 'r', encoding='utf8') as input_file:
            # Treat every character literally (including quotes).
            rows = csv.reader(input_file, delimiter='\t', quotechar=None)
            ids = it.count(1)
            current_doc_id = 0
            current_sentences = []
            for new_doc, doc_rows in it.groupby(rows, MedMentionsDataset.is_a_document_separator):
                if new_doc:
                    if current_sentences:
                        document = Document(id=current_doc_id, sentences=current_sentences)
                        documents.append(document)
                        current_sentences = []
                        current_doc_id = next(ids)
                else:
                    current_tokens = []
                    for new_sentence, sentence_row in it.groupby(doc_rows,
                                                                 MedMentionsDataset.sentence_separator):
                        if new_sentence:
                            if current_tokens:
                                sentence = Sentence(tokens=current_tokens)
                                current_sentences.append(sentence)
                                current_tokens = []
                        else:
                            for raw_token in sentence_row:
                                token = self.create_annotated_token_from_row(raw_token)
                                encoded_token = self.create_encoded_token_from_token(token, encoder=encoder)
                                current_tokens.append(encoded_token)
                    sentence = Sentence(tokens=current_tokens)
                    current_sentences.append(sentence)
            document = Document(id=current_doc_id, sentences=current_sentences)
            documents.append(document)
        return documents

    @staticmethod
    def is_a_document_separator(row):
        if len(row) == 0:
            return False
        elif row[0].startswith(MedMentionsDataset.DOC_START):
            return True
        else:
            return False

    @staticmethod
    def sentence_separator(row):
        if len(row) == 0:
            return True
        return False

    @staticmethod
    def create_annotated_token_from_row(row):
        assert len(row) == 4

        tag = BIO2Tag(row[3][0])
        return Token(text=row[0], start=row[1], end=row[2], tag=tag)

    @staticmethod
    def create_encoded_token_from_token(token, encoder):
        encoding = encoder[token.text]
        return EncodedToken(encoding=encoding, text=token.text, start=token.start, end=token.end, tag=token.tag)

    def flatten_dataset(self):
        flatten = it.chain.from_iterable
        self.sentences = []
        for sentence in flatten(self.documents):
            tokens = []
            for token in sentence.tokens:
                tokens.append(token)
            self.sentences.append(tokens)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentences = self.sentences[index]
        encodings = []
        tags = []
        for token in sentences:
            encodings.append(token.encoding)
            tags.append(BIO2Tag.get_index(token.tag))
        return encodings, tags
