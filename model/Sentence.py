class Sentence:
    def __init__(self, tokens=None):
        if tokens is None:
            tokens = []
        self.tokens = tokens

    def __iter__(self):
        return iter(self.tokens)

    def __eq__(self, other):
        if isinstance(other, Sentence):
            return self.tokens == other.tokens
        return False
