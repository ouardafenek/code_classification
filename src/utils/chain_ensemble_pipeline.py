from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier

from typing import List

import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# =====================
# Pipeline wrapper
# =====================

class ChainEnsemblePipeline:
    """
    Wrapper behaving like a sklearn Pipeline:
    TF-IDF vectorizer + ensemble of Classifier Chains
    """

    def __init__(
        self,
        tfidf: TfidfVectorizer,
        chains: List[ClassifierChain],
        threshold: float = 0.5,
    ):
        self.tfidf = tfidf
        self.chains = chains
        self.threshold = threshold

    def predict_proba(self, X):
        X_vec = self.tfidf.transform(X)
        probs_sum = None

        for chain in self.chains:
            probs = chain.predict_proba(X_vec)
            probs_sum = probs if probs_sum is None else probs_sum + probs

        return probs_sum / len(self.chains)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)
