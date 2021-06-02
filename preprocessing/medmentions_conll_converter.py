import bconv
import argparse
import pathlib
from enum import Enum, auto


class DatasetType(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    TEST = auto()


def convert_medmentions_dataset_to_conll_format(input_filepath):
    """
    Converts the MedMentions dataset from the PubTator format to the CoNLL format.

    The MedMentions dataset is by default in the PubTator format.
    For details about the PubTator format see: https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/PubTator/tutorial/index.html#ExportannotationinPubTator
    For details about the CoNLL format see: https://github.com/lfurrer/bconv/wiki/CoNLL
    :param input_filepath: Path to the original MedMentions dataset in the PubTator format
    """
    corpus_folder_path = pathlib.Path(input_filepath).parent
    output_filepath = corpus_folder_path.joinpath("corpus_CoNLL.txt")
    medmentions = bconv.load(input_filepath, fmt='pubtator')
    with open(output_filepath, 'w', encoding='utf8') as output_file:
        bconv.dump(medmentions, output_file, fmt='conll')


def create_conll_training_validation_test_datasets(corpus_folder_path):
    corpus_conll_filepath = corpus_folder_path.joinpath("corpus_pubtator_CoNLL.txt")
    training_dataset_filepath = corpus_folder_path.joinpath("training_CoNLL.txt")
    validation_dataset_filepath = corpus_folder_path.joinpath("validation_CoNLL.txt")
    test_dataset_filepath = corpus_folder_path.joinpath("test_CoNLL.txt")

    training_ids_filepath = corpus_folder_path.joinpath("corpus_pubtator_pmids_trng.txt")
    validation_ids_filepath = corpus_folder_path.joinpath("corpus_pubtator_pmids_dev.txt")
    test_ids_filepath = corpus_folder_path.joinpath("corpus_pubtator_pmids_test.txt")

    training_ids = parse_ids(training_ids_filepath)
    validation_ids = parse_ids(validation_ids_filepath)
    test_ids = parse_ids(test_ids_filepath)
    current_dataset_type = None
    with open(corpus_conll_filepath, 'r') as corpus_file, \
            open(training_dataset_filepath, 'w', encoding='utf8') as training_output_file, \
            open(validation_dataset_filepath, 'w', encoding='utf8') as validation_output_file, \
            open(test_dataset_filepath, 'w', encoding='utf8') as test_output_file:
        for line in corpus_file:
            index = line.find("# doc_id = ")
            if index != -1:
                index += 11  # add the length of the string "# doc_id = "
                current_doc_id = line[index:].rstrip()
                if current_doc_id in training_ids:
                    current_dataset_type = DatasetType.TRAINING
                elif current_doc_id in validation_ids:
                    current_dataset_type = DatasetType.VALIDATION
                elif current_doc_id in test_ids:
                    current_dataset_type = DatasetType.TEST
                else:
                    raise RuntimeError(
                        "Document ID: " + current_doc_id + " not found in the training, validation, or test set.")
            else:
                if current_dataset_type is None:
                    continue
                if current_dataset_type == DatasetType.TRAINING:
                    training_output_file.write(line)
                elif current_dataset_type == DatasetType.VALIDATION:
                    validation_output_file.write(line )
                elif current_dataset_type == DatasetType.TEST:
                    test_output_file.write(line)
                else:
                    raise RuntimeError("Invalid DatasetType")


def parse_ids(filepath):
    ids = []
    with open(filepath, 'r') as file:
        for line in file:
            if line:
                ids.append(line.rstrip())
    return ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MedMentions dataset from PubTator to CoNLL')
    parser.add_argument('input_filepath',
                        metavar='input_filepath',
                        type=str,
                        help='Path to the original MedMentions dataset in the PubTator format')
    args = parser.parse_args()
    convert_medmentions_dataset_to_conll_format(input_filepath=args.input_filepath)
    corpus_folder_path = pathlib.Path(args.input_filepath).parent
    create_conll_training_validation_test_datasets(corpus_folder_path=corpus_folder_path)
