import argparse
import csv
from copy import deepcopy

from bioner.model.bio2tag import BIO2Tag
from bioner.model.conll_dataset import CoNLLDataset


def read_prediction_tsv_file(path: str):
    annotated_tokens = []
    with open(path) as tsv_file:
        reader = csv.reader(tsv_file, delimiter=" ", quotechar=None)
        for row in reader:
            if len(row) == 2:
                token = row[0]
                label = row[1]
                annotated_tokens.append([token, label])
    return annotated_tokens


def annotate(dataset: CoNLLDataset, predictions_file_path: str) -> CoNLLDataset:
    annotated_tokens = read_prediction_tsv_file(path=predictions_file_path)
    annotated_dataset = deepcopy(dataset)
    i = 0
    for doc_index, document in enumerate(dataset.documents):
        for sentence_index, sentence in enumerate(document.sentences):
            for token_index, token in enumerate(sentence.tokens):
                document_to_annotate = annotated_dataset.documents[doc_index]
                sentence_to_annotate = document_to_annotate.sentences[sentence_index]
                token = sentence_to_annotate.tokens[token_index]
                annotated_token = annotated_tokens[i]
                if token.text != annotated_token[0]:
                    raise ValueError(f"Expected Token: {token.text} actual token: {annotated_token[0]}")
                assert token.text == annotated_token[0]
                label = BIO2Tag(annotated_token[1])
                token.tag = label
                i += 1
    return annotated_dataset


def annotate_to_file(dataset_file_path: str, predictions_file_path: str, output_file_path: str):
    dataset = CoNLLDataset(data_file_path=dataset_file_path)
    annotated_dataset = annotate(dataset=dataset, predictions_file_path=predictions_file_path)
    CoNLLDataset.write_dataset_to_file(dataset=annotated_dataset, file_path=output_file_path)


def write_dataset_to_conll_file(dataset: CoNLLDataset, annotated_dataset: CoNLLDataset, file_path: str):
    with open(file_path, 'w', encoding='utf8') as output_file:
        for document_index, document in enumerate(dataset.documents):
            annotated_document = annotated_dataset.documents[document_index]
            for sentence_index, sentence in enumerate(document.sentences):
                annotated_sentence = annotated_document.sentences[sentence_index]
                output_file.write("\n")
                for token_index, token in enumerate(sentence.tokens):
                    annotated_token = annotated_sentence.tokens[token_index]
                    token_str = f"{token.text} {token.tag.value} {annotated_token.tag.value}"
                    output_file.write(token_str + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate BioBERT prediction file')
    parser.add_argument("--evaluationDatasetPath", required=True)
    parser.add_argument("--predictionsFilePath", required=True)
    parser.add_argument("--outputFilePath", required=True)
    args = parser.parse_args()
    annotate_to_file(dataset_file_path=args.evaluationDatasetPath,
                     predictions_file_path=args.predictionsFilePath,
                     output_file_path=args.outputFilePath)
