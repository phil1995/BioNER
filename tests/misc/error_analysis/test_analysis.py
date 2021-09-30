import unittest
from copy import deepcopy

from bioner.misc.error_analysis.analysis import select_errors, ManualErrorAnalysis, ErrorStatistics, ErrorAnalysis, \
    calc_overlapping_statistics
from bioner.model.BIO2Tag import BIO2Tag
from bioner.model.CoNLLDataset import CoNLLDataset
from bioner.model.metrics.EntityLevelPrecisionRecall import Annotation


def test_export_to_csv(tmpdir):
    output_file_path = tmpdir.join("output.csv")

    gold_dataset = create_gold_dataset(tmpdir)
    annotated_dataset = deepcopy(gold_dataset)

    # Introduce error: tag "ipsum" as outside
    annotated_dataset.documents[0].sentences[0].tokens[1].tag = BIO2Tag.OUTSIDE

    # Check that gold_dataset has still the correct tag
    assert gold_dataset.documents[0].sentences[0].tokens[1].tag == BIO2Tag.INSIDE

    analysis = select_errors(gold_standard_dataset=gold_dataset, dataset=annotated_dataset, seed=1632737901)
    ManualErrorAnalysis.export_to_csv(error_analysis_objects=[analysis], output_file_path=output_file_path)

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


def test_annotation_length_error_statistics(tmpdir):
    gold_dataset = create_gold_dataset(tmpdir)

    annotated_dataset = deepcopy(gold_dataset)

    # Introduce error: tag "ipsum" as outside
    annotated_dataset.documents[0].sentences[0].tokens[1].tag = BIO2Tag.OUTSIDE

    # Check that gold_dataset has still the correct tag
    assert gold_dataset.documents[0].sentences[0].tokens[1].tag == BIO2Tag.INSIDE

    error_statistic = ErrorStatistics(gold_standard_dataset=gold_dataset)
    statistic_result = error_statistic.calc_error_stats_for_lengths(annotated_dataset)

    assert statistic_result.errors == {2: 1}
    assert statistic_result.total_annotations == {1: 2, 2: 1}


def test_error_analysis_annotations(tmpdir):
    gold_dataset = create_gold_dataset(tmpdir)

    annotated_dataset = deepcopy(gold_dataset)

    # Introduce error: tag "ipsum" as outside
    annotated_dataset.documents[0].sentences[0].tokens[1].tag = BIO2Tag.OUTSIDE

    # Check that gold_dataset has still the correct tag
    assert gold_dataset.documents[0].sentences[0].tokens[1].tag == BIO2Tag.INSIDE

    error_analysis = ErrorAnalysis(gold_standard_dataset=gold_dataset, dataset=annotated_dataset)
    assert len(error_analysis.false_negative_annotations) == 0
    assert len(error_analysis.false_positive_annotations) == 0
    assert len(error_analysis.true_positive_annotations) == 0

    error_analysis.analyze_annotations()

    assert error_analysis.false_negative_annotations == [Annotation(sentence_id=0, start_token_id=0, end_token_id=1)]
    assert error_analysis.false_positive_annotations == [Annotation(sentence_id=0, start_token_id=0, end_token_id=0)]
    assert error_analysis.true_positive_annotations == [Annotation(sentence_id=1, start_token_id=0, end_token_id=0),
                                                        Annotation(sentence_id=2, start_token_id=0, end_token_id=0)]


def test_error_analysis_overlapping(tmpdir):
    gold_dataset = create_gold_dataset(tmpdir)

    annotated_dataset_1 = deepcopy(gold_dataset)

    # Introduce error: tag "ipsum" as outside
    annotated_dataset_1.documents[0].sentences[0].tokens[1].tag = BIO2Tag.OUTSIDE

    annotated_dataset_2 = deepcopy(annotated_dataset_1)

    # Introdcue another error: tag "ut" as outside
    annotated_dataset_2.documents[1].sentences[0].tokens[0].tag = BIO2Tag.OUTSIDE

    # Check that gold_dataset has still the correct tag
    assert gold_dataset.documents[0].sentences[0].tokens[1].tag == BIO2Tag.INSIDE

    error_analysis_1 = ErrorAnalysis(gold_standard_dataset=gold_dataset, dataset=annotated_dataset_1)
    error_analysis_1.analyze_annotations()
    error_analysis_2 = ErrorAnalysis(gold_standard_dataset=gold_dataset, dataset=annotated_dataset_2)
    error_analysis_2.analyze_annotations()
    result = calc_overlapping_statistics(analysis_1=error_analysis_1,
                                         analysis_2=error_analysis_2)
    assert result.true_positives == 1
    assert result.false_negatives == 1
    assert result.false_positives == 1


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


def create_gold_dataset(tmpdir) -> CoNLLDataset:
    file_path = tmpdir.join("test_CoNLL_file.txt")
    content = create_test_document_content()
    with open(file_path, "w") as text_file:
        text_file.write(content)
    return CoNLLDataset(file_path)
