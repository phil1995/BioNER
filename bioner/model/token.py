from typing import Optional

from bioner.model.bio2tag import BIO2Tag


class Token:
    def __init__(self, text: str, start: str, end: str, tag: Optional[BIO2Tag]):
        """
        :param text: the text which represents the token
        :param start: the start position of the token
        :param end: the end position of the token (= start position + text length)
        :param tag: the optional BIO2 tag assigned to the token
        """
        self.text = text
        self.start = start
        self.end = end
        self.tag = tag

    def __str__(self):
        tag_str = self.tag.name if self.tag is not None else "not set"
        return f"{self.text} - start: {self.start} - end: {self.end} - tag: {tag_str}"

    def __eq__(self, other):
        if isinstance(other, Token):
            return (
                self.text == other.text
                and self.start == other.start
                and self.end == other.end
                and self.tag == other.tag
            )

        return False
