import os
import numpy as np
from nltk.corpus.reader import CorpusReader
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from typing import List
import math


class CorpusReader_TFIDF:
    """
    Args
     - corpus: CorpusReader object
     - tf: "raw" or "log" (default: "raw")
     - idf: "base" or "smooth"(default: "base")
     - stopword:
        - "none"(no stopwords, default),
        - "standard"(standard English stopWord in NLTK)
        - provide a filename for custom stopwords (file should be located in the current directory)
        - to_stem: Boolean, if True, Snowball stemmer is used.(default: False)
        - stem_first: Boolean, if True, stem before stopword removal(default: False)
        - ignore_case: Boolean, if False, no case normalization (default: True)
    """

    def __init__(self, corpus: CorpusReader, tf="raw", idf="base", stopword="none", to_stem=False, stem_first=False, ignore_case=True) -> None:
        # Initialize parameters
        self.corpus = corpus
        self.tf = tf
        self.idf = idf
        self.stopword = stopword
        self.to_stem = to_stem
        self.stem_first = stem_first
        self.ignore_case = ignore_case

        # Define valid parameter options
        self.current_dir = os.getcwd()
        tf_methods = ["raw", "log"]
        idf_methods = ["base", "smooth"]

        # Validate TF and IDF mothods
        if tf not in tf_methods:
            raise ValueError(
                f"Invalid value for tf: {tf}. Must be one of {tf_methods}")
        if idf not in idf_methods:
            raise ValueError(
                f"Invalid value for idf: {idf}. Must be one of {idf_methods}")

        # Handle stopword options
        if stopword == "none":
            self.stops = set()
        elif stopword == "standard":
            self.stops = set(stopwords.words('english'))
        else:
            path = os.path.join(self.current_dir, stopword)
            try:
                with open(path, 'r') as stopword_file:
                    self.stops = set(stopword_file.read().split())
            except FileNotFoundError:
                raise ValueError(f"Stopword file not found:{stopword}")
        # Initialize stemmer
        if to_stem == True:
            self.stemmer = SnowballStemmer('english')

        # Preload the corpus
        self.docs_set = {fileid: self.corpus.words(
            fileid) for fileid in self.corpus.fileids()}

    def _calculate_tf(self, term, fileid):
        words = self.docs_set[fileid]
        frequency = words.count(term)
        if self.tf == "log":
            return 1 + math.log2(frequency) if frequency > 0 else 0
        else:
            return frequency

    def _calculate_idf(self, term):
        num_of_docs = len(self.docs_set)
        docs_with_term = 0
        for doc in self.docs_set:
            if term in self.docs_set[doc]:
                docs_with_term += 1
        if docs_with_term == 0:
            return 1e-6
        if self.idf == "base":
            return math.log2(num_of_docs / docs_with_term)
        else:
            return math.log2(1 + (num_of_docs / docs_with_term))

    def _preprocess(self, words: List):
        normalized_words = [word.casefold()
                            for word in words] if self.ignore_case == True else words
        stopword_filtered = [
            word for word in normalized_words if word not in self.stops]
        if self.to_stem == True:
            stemmed_words = [self.stemmer.stem(
                word) for word in stopword_filtered]
            return stemmed_words
        else:
            return stopword_filtered

    def tf_idf(self, fileid, return_zero=False):
        """
        Calculate TF-IDF for a document identified by fileid
        Args:
            return_zero (bool, optional): If returnZero is True, then the dictionary will contain terms that have 0 value for that vector, otherwise the vector will omit those terms. Defaults to False.

        Returns:
            dict: keys are each term, and values are the corresponding TF-IDF score.
        """
        score = {}
        words = self._preprocess(self.docs_set[fileid])
        unique_terms = set(words)
        for term in unique_terms:
            tf = self._calculate_tf(term,  fileid)
            idf = self._calculate_idf(term)
            tfidf = tf * idf
            if not return_zero and tfidf == 0:
                continue
            score[term] = tfidf
        return score

    def tf_idf_all(self, return_zero=False):
        """
        Calculate TF-IDF for all the documents in the corpus
        Args:
            return_zero (bool, optional): If returnZero is True, then the dictionary will contain terms that have 0 value for that vector, otherwise the vector will omit those terms. Defaults to False.

        Returns:
            dict: keys are fileid of each document, and values are the corresponding TF-IDF scores of the document.
        """
        score_all = {}
        for fileid in self.docs_set:
            score = self.tf_idf(fileid, return_zero=return_zero)
            if not return_zero and score == 0:
                continue
            score_all[fileid] = score
        return score_all

    def tfidf_new(self, words: List):
        score = {}
        words = self._preprocess(words)
        unique_terms = set(words)
        for term in unique_terms:
            tf = words.count(term)
            if self.tf == "log":
                tf = 1 + math.log2(tf) if tf > 0 else 0
            idf = self._calculate_idf(term)
            tfidf = tf * idf
            score[term] = tfidf
        return score

    def idf_scores_all(self):
        """
        Return the IDF of each term in the corpus as a dictionary
        Returns:

            dict: keys are the terms, and the values are the IDF
        """
        idf_scores = {}
        all_terms = [term for fileid in self.docs_set for term in self._preprocess(
            self.docs_set[fileid])]
        unique_terms = set(all_terms)
        for term in unique_terms:
            idf = self._calculate_idf(term)
            idf_scores[term] = idf
        return idf_scores

    def cosine_sim(self, fileid_1, fileid_2):
        tfidf_A = self.tf_idf(fileid_1)
        tfidf_B = self.tf_idf(fileid_2)

        # Equalize dimensions
        all_terms = list(tfidf_A.keys())
        for term in tfidf_B.keys():
            if term not in all_terms:
                all_terms.append(term)

        # Convert to numpy array
        vector_A = np.array([tfidf_A.get(term, 0) for term in all_terms])
        vector_B = np.array([tfidf_B.get(term, 0) for term in all_terms])

        cosine_sim = np.dot(vector_A, vector_B) / \
            (np.linalg.norm(vector_A) * np.linalg.norm(vector_B))

        return cosine_sim

    def cosine_sim_new(self, words: List, fileid):
        tfidf_new = self.tfidf_new(words)
        tfidf_original = self.tf_idf(fileid)

        # Equalize dimensions
        all_terms = list(tfidf_new.keys())
        for term in tfidf_original.keys():
            if term not in all_terms:
                all_terms.append(term)

        # Convert to numpy array
        vector_A = np.array([tfidf_new.get(term, 0) for term in all_terms])
        vector_B = np.array([tfidf_original.get(term, 0)
                            for term in all_terms])

        cosine_sim = np.dot(vector_A, vector_B) / \
            (np.linalg.norm(vector_A) * np.linalg.norm(vector_B))

        return cosine_sim

    def query(self, words: List):
        """
        Calculate the cosine similarity between the new document(list of words) and the corpus documents

        Returns:
          A list of tuples(document, cosine_similarity), sorted in descending order
        """
        query_result = [(fileid, self.cosine_sim_new(words, fileid))
                        for fileid in self.docs_set]
        query_result.sort(key=lambda a: a[1], reverse=True)
        return query_result
