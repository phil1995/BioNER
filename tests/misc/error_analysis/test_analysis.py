import unittest
from copy import deepcopy

from bioner.misc.error_analysis.analysis import select_errors, ErrorAnalysis
from bioner.model.BIO2Tag import BIO2Tag
from bioner.model.CoNLLDataset import CoNLLDataset


def test_export_to_csv(tmpdir):
    file_path = tmpdir.join("test_CoNLL_file.txt")
    output_file_path = tmpdir.join("output.csv")
    content = create_test_document_content()
    with open(file_path, "w") as text_file:
        text_file.write(content)
    gold_dataset = CoNLLDataset(file_path, encoder=None)
    annotated_dataset = deepcopy(gold_dataset)

    # Introduce error: tag "ipsum" as outside
    annotated_dataset.documents[0].sentences[0].tokens[1].tag = BIO2Tag.OUTSIDE

    # Check that gold_dataset has still the correct tag
    assert gold_dataset.documents[0].sentences[0].tokens[1].tag == BIO2Tag.INSIDE

    analysis = select_errors(gold_standard_dataset=gold_dataset, dataset=annotated_dataset, seed=1632737901)
    ErrorAnalysis.export_to_csv(error_analysis_objects=[analysis], output_file_path=output_file_path)

    expected_lines = ['Text,-1,Lorem,ipsum,dolor\n',
                      'Gold Standard,-1,B,I,O\n',
                      'BioNER,1,B,O,O\n',
                      '\n',
                      'Text,-1,Eirmod,tempor,.\n',
                      'Gold Standard,-1,B,O,O\n',
                      'BioNER,0,B,O,O\n',
                      '\n',
                      'Text,-1,ut,labore,et\n',
                      'Gold Standard,-1,B,O,O\n',
                      'BioNER,0,B,O,O\n',
                      '\n',
                      'Text,-1,dolore,magna,aliquyam\n',
                      'Gold Standard,-1,O,O,O\n',
                      'BioNER,0,O,O,O\n',
                      '\n',
                      ]
    with open(output_file_path, 'r') as exported_csv:
        for index, line in enumerate(exported_csv):
            assert line == expected_lines[index]



# TODO: Refactor Helper --> Duplicate!
# Helper
def create_test_document_content() -> str:
    return """-DOCSTART-	0	0	O

Lorem	0	5	B-T116,T123
ipsum	6	10	I-T047
dolor	11	16	O

Eirmod	0	5	B-T116,T123
tempor	6	8	O
.	9	10	O

-DOCSTART-	0	0	O

ut	0	5	B-UnknownType
labore	6	8	O
et	9	10	O

dolore	0	5	O
magna	6	8	O
aliquyam	9	10	O
"""
