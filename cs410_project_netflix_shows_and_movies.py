# TF-IDF / Similarity Score Computation Pipeline

import numpy as np
import pandas as pd
import math
from collections import Counter
from flask import Flask, request, jsonify
import ast  

# Loading cleaned dataset
df_revised = pd.read_csv("netflix_cleaned.csv")  # columns: title, description, tokenized_description, rating
df_revised['tokenized_description'] = df_revised['tokenized_description'].apply(ast.literal_eval)

class TFIDFSearch:
  def __init__(self, dataset):
        print("Initializing TFIDFSearch...")
        self.dataset = pd.DataFrame(dataset, columns=["title", "tokenized_description"])
        print(f"Dataset size: {len(self.dataset)} documents")
        self.build_vocabulary()
        self.IDF = None

  '''
    Build vocabulary from the dataset
  '''
  def build_vocabulary(self):
    print("Building vocabulary...")
    freq = Counter()
    for _, row in self.dataset.iterrows():
        tokens = row["tokenized_description"]
        if isinstance(tokens, list):
            freq.update(tokens)
    self.vocab = np.array([word for word, _ in freq.most_common(200)])
    print(f"Vocabulary built: {len(self.vocab)} words")


  '''
    Compute IDF values for the vocabulary
  '''
  def compute_IDF(self):
    print("Computing IDF values...")
    collection = self.dataset["tokenized_description"]
    self.IDF = np.zeros(self.vocab.size) # Initialize the IDFs to zero
    M = len(self.dataset)
    
    # building document frequency dict once 
    doc_freq = Counter()
    for doc in collection:
      unique_words = set(doc)  # only counting each word once per document
      doc_freq.update(unique_words)
    
    # computing IDF for each word in vocab
    for i, word in enumerate(self.vocab):
      k = doc_freq.get(word, 0)  
      if k == 0:
        k = 1
      self.IDF[i] = math.log((M + 1) / (k))
    
    return self.IDF

  '''
    Convert text to TF-IDF vector
  '''
  def text2TFIDF(self, text, applyBM25_and_IDF=False):
    vocab = self.vocab
    tfidfVector = np.zeros(vocab.size)

    words = pd.Series(text)
    counts = words.value_counts()

    for i, word in enumerate(vocab):
      if word in text:
        entry = counts.get(word, 0)
        if applyBM25_and_IDF:
          idf = self.IDF[i]
          entry = entry * idf
        tfidfVector[i] = entry
    
    return tfidfVector

  '''
    Compute TF-IDF similarity score between query and document
  '''
  def tfidf_score(self, query, doc, applyBM25_and_IDF=False):
    q = self.text2TFIDF(query)
    d = self.text2TFIDF(doc, applyBM25_and_IDF)
    relevance = np.dot(q, d)

    return relevance

  '''
    Execute search over the dataset for the given query
  '''
  def execute_search_TF_IDF(self, query):
    print(f"executing search for query: {query}")
    relevances = np.zeros(self.dataset.shape[0]) # Initialize relevances of all documents to 0

    for index, row in self.dataset.iterrows():
      doc = row["tokenized_description"]
      relevance = self.tfidf_score(query, doc, applyBM25_and_IDF=True)
      relevances[index] = relevance

    print(f"search complete. found {np.sum(relevances > 0)} relevant documents")
    return relevances # in the same order of the documents in the dataset