from bioner.model.NGramEncoder import NGramEncoder, keep_only_printable_chars, TrigramEncoder


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
