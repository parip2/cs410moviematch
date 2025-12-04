import pandas as pd
import numpy as np
import math

# --- Load cleaned dataset ---
# Make sure you have a CSV of the cleaned data locally, e.g. "netflix_cleaned.csv"
df_revised = pd.read_csv("netflix_cleaned.csv")  # columns: title, description, tokenized_description, rating

# --- TFIDFSearch class ---
class TFIDFSearch:
    def __init__(self, dataset):
        self.dataset = pd.DataFrame(dataset, columns=["title", "tokenized description"])
        self.build_vocabulary()
        self.IDF = None

    def build_vocabulary(self):
        freq = pd.Series(dtype=int)
        for index, row in self.dataset.iterrows():
            words = pd.Series(row["tokenized description"])
            counts = words.value_counts()
            freq = freq.add(counts, fill_value=0)
        self.vocab = np.array(freq.nlargest(200).index.to_list())

    def compute_IDF(self):
        collection = self.dataset["tokenized description"]
        self.IDF  = np.zeros(self.vocab.size)
        M = len(self.dataset)
        for i, word in enumerate(self.vocab):
            k = sum(1 for doc in collection if word in doc)
            if k == 0: k = 1
            self.IDF[i] = math.log((M + 1) / k)
        return self.IDF

    def text2TFIDF(self, text, applyBM25_and_IDF=False):
        tfidfVector = np.zeros(self.vocab.size)
        for i, word in enumerate(self.vocab):
            counts = pd.Series(text).value_counts()
            if word in text:
                entry = counts.get(word, 0)
                if applyBM25_and_IDF:
                    entry *= self.IDF[i]
                tfidfVector[i] = entry
        return tfidfVector

    def tfidf_score(self, query, doc, applyBM25_and_IDF=False):
        q = self.text2TFIDF(query)
        d = self.text2TFIDF(doc, applyBM25_and_IDF)
        return float(np.dot(q, d))

    def execute_search_TF_IDF(self, query):
        self.compute_IDF()
        relevances = np.zeros(self.dataset.shape[0])
        for idx, row in self.dataset.iterrows():
            relevances[idx] = self.tfidf_score(query, row["tokenized description"], applyBM25_and_IDF=True)
        return relevances
