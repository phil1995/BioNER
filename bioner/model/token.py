from typing import Optional

from bioner.model.bio2tag import BIO2Tag


class Token:
    def __init__(self, text: str, start: str, end: str, tag: Optional[BIO2Tag]):
        self.text = text
        self.start = start
        self.end = end
        self.tag = tag

    def __str__(self):
        return f"{self.text} - start: {self.start} - end: {self.end} - tag: {self.tag.name}"

    def __eq__(self, other):
        if isinstance(other, Token):
            result = self.text == other.text and self.start == other.start and self.end == other.end and self.tag == other.tag
            return result
        return False
