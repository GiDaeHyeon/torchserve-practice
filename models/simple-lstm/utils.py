import string
from typing import Iterable
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class SimpleTokenizer:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.dictionary = defaultdict(int)
        self.dictionary.__setitem__('Unknown', 1)
        self.dictionary.__setitem__('PAD', 2)
        self.stemmer = PorterStemmer()

    @staticmethod
    def preprocessing(text: str, **kwargs) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str, **kwargs) -> list:
        preprocessed_text = self.preprocessing(text)
        return [self.stemmer.stem(word) for word in word_tokenize(preprocessed_text)]

    def fit(self, corpus: Iterable, **kwargs) -> None:
        words = []
        for c in corpus:
            preprocessed_text = self.preprocessing(c)
            tokens = self.tokenize(preprocessed_text)
            words += tokens
        words = set(words)

        for idx, w in enumerate(words):
            self.dictionary.__setitem__(w, idx + 3)

    def get_words_num(self) -> int:
        return len(self.dictionary.keys())

    def __call__(self, text: str, **kwargs) -> list:
        return [self.dictionary.get(token, 1) for token in self.tokenize(text)]
