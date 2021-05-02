from preprocessing.pubmed_parser import PubMedParser

if __name__ == '__main__':
    parser = PubMedParser()
    parser.parse_all_files_from("/tests/ressources", combined_file_target_path="/foo")

