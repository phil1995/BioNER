from typing import Optional

from model.BIO2Tag import BIO2Tag


class Token:
    def __init__(self, text: str, start: int, end: int, tag: Optional[BIO2Tag]):
        self.text = text
        self.start = start
        self.end = end
        self.tag = tag

    def __str__(self):
        return f"{self.text} - start: {self.start} - end: {self.end} - tag: {self.tag.name}"
