from typing import Callable
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
from nltk.translate.meteor_score import meteor_score

class METEOR:
    def __init__(
        self, 
        preprocess: Callable[[str], str] = str.lower,
        stemmer: StemmerI = PorterStemmer(),
        wordnet: WordNetCorpusReader = wordnet,
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5
    ):
        self.preprocess = preprocess
        self.stemmer = stemmer
        self.wordnet = wordnet
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.num_hyp = 0
        self.sum_scores = 0.

    def add(self, candidate, refs):
        self.num_hyp += 1
        self.sum_scores += meteor_score(
            refs, 
            candidate, 
            self.preprocess, 
            self.stemmer, 
            self.wordnet, 
            self.alpha, 
            self.beta, 
            self.gamma
        )

    def compute(self):
        return self.sum_scores / self.num_hyp