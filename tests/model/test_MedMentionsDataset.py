from model.BIO2Tag import BIO2Tag
from model.Document import Document
from model.MedMentionsDataset import MedMentionsDataset
from model.Sentence import Sentence
from model.Token import Token


def test_read_documents_empty_lines_at_the_end(tmpdir):
    file_path = tmpdir.join("test_CoNLL_file.txt")
    content = create_test_document_content()
    content += "\n\n"

    with open(file_path, "w") as text_file:
        text_file.write(content)

    dataset = MedMentionsDataset(file_path, encoder=None)
    assert len(dataset.documents) == 2
    expected_documents = create_expected_documents_for_test_document()
    assert expected_documents == dataset.documents


def test_read_documents(tmpdir):
    file_path = tmpdir.join("test_CoNLL_file.txt")
    content = create_test_document_content()

    with open(file_path, "w") as text_file:
        text_file.write(content)

    dataset = MedMentionsDataset(file_path, encoder=None)
    assert len(dataset.documents) == 2
    expected_documents = create_expected_documents_for_test_document()
    assert expected_documents == dataset.documents


def test_flatten_documents(tmpdir):
    file_path = tmpdir.join("test_CoNLL_file.txt")
    content = create_test_document_content()

    with open(file_path, "w") as text_file:
        text_file.write(content)
    dataset = MedMentionsDataset(file_path, encoder=None)

    assert dataset.sentences is None
    dataset.flatten_dataset()
    assert dataset.sentences is not None
    expected_sentences = create_expected_tokens_for_test_document()
    assert expected_sentences == dataset.sentences


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


def create_expected_documents_for_test_document() -> [Document]:
    expected_sentences = create_expected_tokens_for_test_document()
    expected_doc_0 = Document(id=0, sentences=[Sentence(tokens=expected_sentences[0]),
                                               Sentence(tokens=expected_sentences[1]),
                                               ])

    expected_doc_1 = Document(id=1, sentences=[Sentence(tokens=expected_sentences[2]),
                                               Sentence(tokens=expected_sentences[3]),
                                               ])
    return [expected_doc_0, expected_doc_1]


def create_expected_tokens_for_test_document() -> [[Token]]:
    return [[Token(text="Lorem", start='0', end='5', tag=BIO2Tag.BEGIN),
             Token(text="ipsum", start='6', end='10', tag=BIO2Tag.INSIDE),
             Token(text="dolor", start='11', end='16', tag=BIO2Tag.OUTSIDE),
             ],
            [Token(text="Eirmod", start='0', end='5', tag=BIO2Tag.BEGIN),
             Token(text="tempor", start='6', end='8', tag=BIO2Tag.OUTSIDE),
             Token(text=".", start='9', end='10', tag=BIO2Tag.OUTSIDE),
             ],
            [Token(text="ut", start='0', end='5', tag=BIO2Tag.BEGIN),
             Token(text="labore", start='6', end='8', tag=BIO2Tag.OUTSIDE),
             Token(text="et", start='9', end='10', tag=BIO2Tag.OUTSIDE),
             ],
            [Token(text="dolore", start='0', end='5', tag=BIO2Tag.OUTSIDE),
             Token(text="magna", start='6', end='8', tag=BIO2Tag.OUTSIDE),
             Token(text="aliquyam", start='9', end='10', tag=BIO2Tag.OUTSIDE),
             ],
            ]
