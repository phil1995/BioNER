from typing import Optional
from model.BIO2Tag import BIO2Tag
from model.Token import Token


class EncodedToken(Token):
    def __init__(self, encoding, text: str, start: int, end: int, tag: Optional[BIO2Tag]):
        self.encoding = encoding
        super().__init__(text=text, start=start, end=end, tag=tag)

    def __str__(self):
        return super().__str__() + f" - encoding: {self.encoding}"
