from enum import Enum


class BIO2Tag(Enum):
    BEGIN = "B"
    INSIDE = "I"
    OUTSIDE = "O"

    def __str__(self):
        return str(self.value)

    @classmethod
    def get_index(cls, type):
        return list(cls).index(type)

    @classmethod
    def index_to_type(cls, index):
        return list(cls)[index]
