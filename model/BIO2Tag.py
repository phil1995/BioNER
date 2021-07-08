from enum import Enum

class BIO2Tag(Enum):
    BEGIN = "B"
    INSIDE = "I"
    OUTSIDE = "O"

    @classmethod
    def get_index(cls, type):
        return list(cls).index(type)
