import argparse

from bioner.model.CoNLLDataset import CoNLLDataset


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


def main():
    parser = argparse.ArgumentParser(description='Convert CoNLLDataset file to a CoNLL file'
                                                 'for the original eval script')
    parser.add_argument("--goldStandardDatasetFilePath", required=True)
    parser.add_argument("--annotatedDatasetFilePath", required=True)
    parser.add_argument("--outputFilePath", required=True)
    args = parser.parse_args()

    gold_standard_dataset = CoNLLDataset(data_file_path=args.goldStandardDatasetFilePath)
    annotated_dataset = CoNLLDataset(data_file_path=args.annotatedDatasetFilePath)
    write_dataset_to_conll_file(dataset=gold_standard_dataset,
                                annotated_dataset=annotated_dataset,
                                file_path=args.outputFilePath)


if __name__ == "__main__":
    main()
