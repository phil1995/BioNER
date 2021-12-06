from typing import Optional
from bioner.model.bio2tag import BIO2Tag
from bioner.model.token import Token


class EncodedToken(Token):
    def __init__(self, encoding, text: str, start: str, end: str, tag: Optional[BIO2Tag]):
        self.encoding = encoding
        super().__init__(text=text, start=start, end=end, tag=tag)

    def __str__(self):
        return super().__str__() + f" - encoding: {self.encoding}"
