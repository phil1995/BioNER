from bioner.model.conll_dataset import CoNLLDataset
from bioner.model.encoder.ngram_encoder import NGramEncoder, keep_only_printable_chars, TrigramEncoder, Vocabulary, LookupCache


def test_alphabet():
    text = "Test"
    assert text.lower() == keep_only_printable_chars(text)

    text = "(ASA)"
    assert text.lower() == keep_only_printable_chars(text)

    text = "Reye's"
    assert text.lower() == keep_only_printable_chars(text)

    text = "acetaminophen/aspirin/pro-caffeine"
    assert text.lower() == keep_only_printable_chars(text)

    text = "11.8%"
    assert text.lower() == keep_only_printable_chars(text)

    text = "1,667"
    assert text.lower() == keep_only_printable_chars(text)

    text = "a b"
    assert "ab" == keep_only_printable_chars(text)

    text = "25 °C (77 °F)"
    assert "25c(77f)" == keep_only_printable_chars(text)

    text = "\"<cite>\""
    assert text == keep_only_printable_chars(text)

    text = "„(Quelle:http://example.com;Datum)“"
    assert text.lower() == keep_only_printable_chars(text)

    text = "§63"
    assert text.lower() == keep_only_printable_chars(text)

    text = "Maßähnliche"
    assert text.lower() == keep_only_printable_chars(text)

    text = "né"
    assert text.lower() == keep_only_printable_chars(text)

    text = "français"
    assert text.lower() == keep_only_printable_chars(text)

    text = "l'amuïssement"
    assert text.lower() == keep_only_printable_chars(text)

    text = "conquête"
    assert text.lower() == keep_only_printable_chars(text)

    text = "?"
    assert text.lower() == keep_only_printable_chars(text)


def test_n_gram_generation():
    n_gram = NGramEncoder(n=3)
    text = "Aspirin"

    n_grams = n_gram.create_n_grams(text)
    assert n_grams == ["#as", "asp", "spi", "pir", "iri", "rin", "in#"]

    trigram_encoder = TrigramEncoder()
    trigrams = trigram_encoder.create_n_grams(text)
    assert trigrams == n_grams

    five_gram = NGramEncoder(n=5)
    text = "cat"
    five_grams = five_gram.create_n_grams(text)
    assert five_grams == ["#cat#"]


def test_n_gram_encoder(tmpdir):
    file_path = tmpdir.join("test_CoNLL_file.txt")
    content = create_test_data()
    with open(file_path, "w") as text_file:
        text_file.write(content)
    dataset = CoNLLDataset(file_path)
    trigram_encoder = TrigramEncoder()
    trigram_encoder.create_encodings(dataset=dataset, min_word_frequency=1)

    assert trigram_encoder.get_embeddings_vector_size() == 23
    assert not trigram_encoder.is_unknown(word="Prime")
    assert trigram_encoder.is_unknown("Kengo")
    vector1 = trigram_encoder.encode("Minister")
    vector2 = trigram_encoder.encode("Mistister")

    assert len(vector1) == 23
    assert len(vector2) == 23

    assert max(vector1) == 1
    assert max(vector2) == 1

    assert sum(vector1) == 8
    assert sum(vector2) == 5


def test_encodings(tmpdir):
    file_path = tmpdir.join("test_CoNLL_file.txt")
    content = create_test_data()
    with open(file_path, "w") as text_file:
        text_file.write(content)
    dataset = CoNLLDataset(file_path)
    trigram_encoder = TrigramEncoder()
    trigram_encoder.create_encodings(dataset=dataset, min_word_frequency=1)

    size = trigram_encoder.get_embeddings_vector_size()

    vector = trigram_encoder.encode("Minister")

    assert len(vector) == size


def test_vocabulary_index(tmpdir):
    vocabulary = Vocabulary()
    vocabulary.increment_word_count("test")
    vocabulary.increment_word_count("tests")
    vocabulary.increment_word_count("testz")

    vocabulary.increment_word_count("tests")
    vocabulary.increment_word_count("tests")
    vocabulary.increment_word_count("testz")

    lookup_cache = LookupCache(vocabulary=vocabulary)

    assert len(lookup_cache) == 3

    assert lookup_cache.get_index_of_word("tests") == 0
    assert lookup_cache.get_index_of_word("testz") == 1
    assert lookup_cache.get_index_of_word("test") == 2


def create_test_data():
    # Taken from https://github.com/sebastianarnold/TeXoo/blob/514860d96decdf3ff6613dfcf0d27d9845ddcf60/texoo-core/src/test/java/de/datexis/encoder/NGramEncoderTest.java#L188-L191
    return """-DOCSTART-	0	0	O
Zaimean	0	5	B
Prime	6	10	I
Minister	11	16	I
Kisto	0	5	B
"""
