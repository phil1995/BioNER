class Document:
    def __init__(self, id: int, sentences=None):
        self.id = id
        if sentences is None:
            sentences = []
        self.sentences = sentences

    def __iter__(self):
        return iter(self.sentences)

    def __eq__(self, other):
        if isinstance(other, Document):
            return self.id == other.id and self.sentences == other.sentences
        return False
