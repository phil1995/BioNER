from __future__ import annotations
import csv
import itertools as it
from torch.utils.data import Dataset

from bioner.model.BIO2Tag import BIO2Tag
from bioner.model.Document import Document
from bioner.model.EncodedToken import EncodedToken
from bioner.model.Sentence import Sentence
from bioner.model.Token import Token


class CoNLLDataset(Dataset):
    """
    The CoNLLDataset with embeddings

    Note: Before using it as a PyTorch.Dataset you should call flatten_dataset()
    This is because the dataset is structured in the following way:
    |-Document:
      |-Sentence
        |-Token (text, tag, embedding)

    to make it possible to shuffle at the document level.
    """
    DOC_START = "-DOCSTART-"

    def __init__(self, data_file_path: str):
        self.documents = self.read_documents(data_file_path)
        self.sentences = None

    def read_documents(self, data_file_path: str):
        documents = []
        with open(data_file_path, 'r', encoding='utf8') as input_file:
            # Treat every character literally (including quotes).
            rows = csv.reader(input_file, delimiter='\t', quotechar=None)
            ids = it.count(1)
            current_doc_id = 0
            current_sentences = []
            for new_doc, doc_rows in it.groupby(rows, CoNLLDataset.is_a_document_separator):
                if new_doc:
                    if current_sentences:
                        document = Document(id=current_doc_id, sentences=current_sentences)
                        documents.append(document)
                        current_sentences = []
                        current_doc_id = next(ids)
                else:
                    current_tokens = []
                    for new_sentence, sentence_row in it.groupby(doc_rows,
                                                                 CoNLLDataset.sentence_separator):
                        if new_sentence:
                            if current_tokens:
                                sentence = Sentence(tokens=current_tokens)
                                current_sentences.append(sentence)
                                current_tokens = []
                        else:
                            for raw_token in sentence_row:
                                token = self.create_annotated_token_from_row(raw_token)
                                current_tokens.append(token)
                    if current_tokens:
                        sentence = Sentence(tokens=current_tokens)
                        current_sentences.append(sentence)
            document = Document(id=current_doc_id, sentences=current_sentences)
            documents.append(document)
        return documents

    @staticmethod
    def is_a_document_separator(row):
        if len(row) == 0:
            return False
        elif row[0].startswith(CoNLLDataset.DOC_START):
            return True
        else:
            return False

    @staticmethod
    def sentence_separator(row):
        return len(row) == 0

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
            tokens = [token for token in sentence.tokens]
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

    @staticmethod
    def write_dataset_to_file(dataset: CoNLLDataset, file_path: str):
        with open(file_path, 'w', encoding='utf8') as output_file:
            for document in dataset.documents:
                output_file.write("-DOCSTART-	0	0	O\n")
                output_file.write("\n")
                for sentence in document.sentences:
                    for token in sentence.tokens:
                        token_str = f"{token.text}\t{token.start}\t{token.end}\t{token.tag.value}"
                        output_file.write(token_str + "\n")
                    output_file.write("\n")
