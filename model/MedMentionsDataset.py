from fasttext.FastText import _FastText
import csv
import itertools as it

from torch.utils.data import Dataset

from model.BIO2Tag import BIO2Tag
from model.Document import Document
from model.EncodedToken import EncodedToken
from model.Sentence import Sentence
from model.Token import Token


class MedMentionsStructuredDataset:
    DOC_START = "-DOCSTART-"
    flatten = it.chain.from_iterable

    def __init__(self, data_file_path: str, encoder: _FastText):
        self.encoder = encoder
        self.documents = self.read_documents(data_file_path)

    def read_documents(self, data_file_path: str):
        documents = []
        with open(data_file_path, 'r', encoding='utf8') as input_file:
            # Treat every character literally (including quotes).
            rows = csv.reader(input_file, delimiter='\t', quotechar=None)
            ids = it.count(1)
            current_doc_id = 0
            current_sentences = []
            for new_doc, doc_rows in it.groupby(rows, MedMentionsStructuredDataset.is_a_document_separator):
                if new_doc:
                    if current_sentences:
                        document = Document(id=current_doc_id, sentences=current_sentences)
                        documents.append(document)
                        current_sentences = []
                        current_doc_id = next(ids)
                else:
                    current_tokens = []
                    for new_sentence, sentence_row in it.groupby(doc_rows,
                                                                 MedMentionsStructuredDataset.sentence_separator):
                        if new_sentence:
                            if current_tokens:
                                sentence = Sentence(tokens=current_tokens)
                                current_sentences.append(sentence)
                                current_tokens = []
                        else:
                            for raw_token in sentence_row:
                                token = self.create_annotated_token_from_row(raw_token)
                                encoded_token = self.create_encoded_token_from_token(token)
                                current_tokens.append(encoded_token)
            document = Document(id=current_doc_id, sentences=current_sentences)
            documents.append(document)
        return documents

    @staticmethod
    def is_a_document_separator(row):
        if len(row) == 0:
            return False
        elif row[0].startswith(MedMentionsStructuredDataset.DOC_START):
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

    def create_encoded_token_from_token(self, token):
        encoding = self.encoder[token.text]
        return EncodedToken(encoding=encoding, text=token.text, start=token.start, end=token.end, tag=token.tag)


class MedMentionsDataset(Dataset):
    def __init__(self, structured_dataset: MedMentionsStructuredDataset):
        self.data = self.get_transformed_dataset(structured_dataset)

    @staticmethod
    def get_transformed_dataset(dataset: MedMentionsStructuredDataset):
        """
                Transforms a MedMentionsDataset object with the structure:
                |-Document:
                  |-Sentence
                    |-Token (text, tag)
                into: elements of
                [[text_j_1, ..., text_j_i], [tag_j_1, ..., tag_j_i]]
                (here for the j-th sentence)
                """
        flatten = it.chain.from_iterable
        rows = []
        for sentence in flatten(dataset.documents):
            encodings = []
            tags = []
            for token in sentence.tokens:
                encodings.append(token.encoding)
                tags.append(BIO2Tag.get_index(token.tag))
            rows.append([encodings, tags])
        return rows


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
