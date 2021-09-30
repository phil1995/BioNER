import regex

import numpy as np
import torch


class SurfaceEncoder:
    #https://github.com/sebastianarnold/TeXoo/blob/514860d96decdf3ff6613dfcf0d27d9845ddcf60/texoo-core/src/main/java/de/datexis/encoder/impl/SurfaceEncoder.java#L25

    @staticmethod
    def get_embedding_vector_size() -> int:
        return 11

    @staticmethod
    def encode(token: str) -> [float]:
        token = token.strip()
        features = [
            SurfaceEncoder.starts_with_uppercase(token),
            SurfaceEncoder.starts_with_lowercase(token),
            SurfaceEncoder.is_all_uppercase(token),
            SurfaceEncoder.is_all_lowercase(token),
            SurfaceEncoder.is_mixed_case(token),
            SurfaceEncoder.is_all_numeric(token),
            SurfaceEncoder.includes_numeric(token),
            SurfaceEncoder.starts_with_numeric(token),
            SurfaceEncoder.ends_with_numeric(token),
            SurfaceEncoder.starts_with_punctuation(token),
            SurfaceEncoder.ends_with_punctuation(token)
        ]
        vector = np.zeros(len(features))
        for i, feature in enumerate(features):
            vector[i] = 1 if feature else 0
        return vector

    @staticmethod
    def starts_with_uppercase(token: str) -> bool:
        token = regex.sub('[^\\p{L}]', '', token)
        if len(token) == 0:
            return False
        return token[0].isupper()

    @staticmethod
    def starts_with_lowercase(token: str) -> bool:
        token = regex.sub('[^\\p{L}]', '', token)
        if len(token) == 0:
            return False
        return token[0].islower()

    @staticmethod
    def is_all_uppercase(token: str) -> bool:
        token = regex.sub('[^\\p{L}]', '', token)
        if len(token) == 0:
            return False
        return token.isupper()

    @staticmethod
    def is_all_lowercase(token: str) -> bool:
        token = regex.sub('[^\\p{L}]', '', token)
        if len(token) == 0:
            return False
        return token.islower()

    @staticmethod
    def is_mixed_case(token: str) -> bool:
        return not SurfaceEncoder.starts_with_uppercase(token) and not SurfaceEncoder.is_all_uppercase(token) \
               and not SurfaceEncoder.is_all_lowercase(token)

    @staticmethod
    def is_all_numeric(token: str) -> bool:
        return token == regex.sub('[^\\p{N}\\p{P}]', '', token)

    @staticmethod
    def includes_numeric(token: str) -> bool:
        return len(regex.sub('[^\\p{N}\\p{P}]', '', token)) != 0

    @staticmethod
    def starts_with_numeric(token: str) -> bool:
        if len(token) == 0:
            return False
        token = token[0]
        return token == regex.sub('[^\\p{N}\\p{P}]', '', token)

    @staticmethod
    def ends_with_numeric(token: str) -> bool:
        if len(token) == 0:
            return False
        token = token[-1]
        return token == regex.sub('[^\\p{N}\\p{P}]', '', token)

    @staticmethod
    def starts_with_punctuation(token: str) -> bool:
        if len(token) == 0:
            return False
        token = token[0]
        return token == regex.sub('[^\\p{P}]', '', token)

    @staticmethod
    def ends_with_punctuation(token: str) -> bool:
        if len(token) == 0:
            return False
        token = token[-1]
        return token == regex.sub('[^\\p{P}]', '', token)

