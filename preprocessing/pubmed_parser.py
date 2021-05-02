import gzip
import os
import xml.etree.ElementTree as ET
import re
import glob
import argparse


class PubMedParser:
    abstract_truncated_at_250_words = "(ABSTRACT TRUNCATED AT 250 WORDS)"
    abstract_truncated_at_400_words = "(ABSTRACT TRUNCATED AT 400 WORDS)"
    abstract_truncated = "(ABSTRACT TRUNCATED)"

    def __init__(self):
        self._abstract_truncated_at_250_words_counter = 0
        self._abstract_truncated_at_400_words_counter = 0
        self._abstract_truncated_counter = 0
        self._abstract_counter = 0

    def parse_pubmed_from(self, source_filepath, target_filepath):
        with gzip.open(source_filepath, 'rb') as source_file, open(target_filepath, "a") as target_file:
            file_content = source_file.read()
            root = ET.fromstring(file_content)

            for article in root.iter('Article'):
                abstract = article.find('Abstract')
                if abstract is None:
                    abstract = article.find('OtherAbstract')
                if abstract is None:
                    continue

                abstract_text = abstract.find('AbstractText')
                if abstract_text is None:
                    continue
                abstract_text_str = abstract_text.text
                if abstract_text_str is None:
                    continue
                target_file.write(self.process_abstract_text(abstract_text_str))
                self._abstract_counter += 1

    def process_abstract_text(self, text):
        processed_text = text
        if text.endswith(PubMedParser.abstract_truncated_at_250_words):
            self._abstract_truncated_at_250_words_counter += 1
            processed_text = re.sub(PubMedParser.abstract_truncated_at_250_words + "$", "", text)
        elif text.endswith(PubMedParser.abstract_truncated_at_400_words):
            self._abstract_truncated_at_400_words_counter += 1
            processed_text = re.sub(PubMedParser.abstract_truncated_at_400_words + "$", "", text)
        elif text.endswith(PubMedParser.abstract_truncated):
            self._abstract_truncated_counter += 1
            processed_text = re.sub(PubMedParser.abstract_truncated + "$", "", text)
        processed_text = processed_text + "\n"
        return processed_text

    def parse_all_files_from(self, folder_path, combined_file_target_path, logfile_path):
        assert not os.path.exists(combined_file_target_path), "A file already exists at the target path - delete this " \
                                                              "file first "
        all_pubmed_archives = [f for f in glob.glob(f"{folder_path}/*.xml.gz")]
        for pubmed_archive_file_path in all_pubmed_archives:
            self.parse_pubmed_from(pubmed_archive_file_path, combined_file_target_path)
        total_truncated_abstracts = self._abstract_truncated_counter + self._abstract_truncated_at_250_words_counter + self._abstract_truncated_at_400_words_counter
        with open(logfile_path, "w") as logfile:
            logfile.writelines([
                f"Total Abstracts: {self._abstract_counter}\n",
                f"Total truncated Abstracts: {total_truncated_abstracts}\n",
                f"Truncated Abstracts at 250 Words: {self._abstract_truncated_at_250_words_counter}\n",
                f"Truncated Abstracts at 400 Words: {self._abstract_truncated_at_400_words_counter}\n",
                f"Truncated Abstracts: {self._abstract_truncated_counter}\n",
            ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess PubMed data')
    parser.add_argument('PubMedFolder',
                        metavar='pubMedFolder',
                        type=str,
                        help='Path of the folder containing the PubMed data')
    parser.add_argument('OutputFile',
                        metavar='outputFile',
                        type=str,
                        help='Path of the file containing the processed abstracts')
    parser.add_argument('LogFile',
                        metavar='logFile',
                            type=str,
                            help='Path to Logfile')
    args = parser.parse_args()
    pubmed_parser = PubMedParser()
    pubmed_parser.parse_all_files_from(folder_path=args.PubMedFolder,
                                       combined_file_target_path=args.OutputFile,
                                       logfile_path=args.LogFile)
