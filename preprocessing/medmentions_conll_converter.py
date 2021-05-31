import bconv
import argparse


def convert_medmentions_dataset_to_conll_format(input_filepath, output_filepath):
    """
    Converts the MedMentions dataset from the PubTator format to the CoNLL format.

    The MedMentions dataset is by default in the PubTator format.
    For details about the PubTator format see: https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/PubTator/tutorial/index.html#ExportannotationinPubTator
    For details about the CoNLL format see: https://github.com/lfurrer/bconv/wiki/CoNLL
    :param input_filepath: Path to the original MedMentions dataset in the PubTator format
    :param output_filepath: Path to the file in which the converted (CoNLL) MedMentions dataset should be saved
    """
    medmentions = bconv.load(input_filepath, fmt='pubtator')
    with open(output_filepath, 'w', encoding='utf8') as output_file:
        bconv.dump(medmentions, output_file, fmt='conll')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MedMentions dataset from PubTator to CoNLL')
    parser.add_argument('input_filepath',
                        metavar='input_filepath',
                        type=str,
                        help='Path to the original MedMentions dataset in the PubTator format')
    parser.add_argument('output_filepath',
                        metavar='output_filepath',
                        type=str,
                        help='Path to the file in which the converted (CoNLL) MedMentions dataset should be saved')
    args = parser.parse_args()
    convert_medmentions_dataset_to_conll_format(input_filepath=args.input_filepath, output_filepath=args.output_filepath)
