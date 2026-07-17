import argparse
import json
import string
import math
import config
from collections import Counter
from nltk.stem import PorterStemmer
from pickle import dump, load
import os

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}


    def __add_document(self, doc_id, text):
        tokenized_text = preprocess(text)
        self.term_frequencies[doc_id] = Counter()
        for token in tokenized_text:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(tokenized_text)

    def get_documents(self, term):
        try:
            document_ids = self.index[term]
        except KeyError:
            return []
        return sorted(document_ids)

    def get_tf(self, doc_id, term):
        try:
            freq = self.term_frequencies[doc_id][term]
            return freq
        except KeyError:
            return 0

    def get_bm25_idf(self, term: str) -> float:
        doc_ids = self.get_documents(term)
        score = math.log((len(self.docmap) - len(doc_ids) + 0.5) / (len(doc_ids) + 0.5) + 1)
        return score

    def get_bm25_tf(self, doc_id, term, k1=config.BM_K1, b=config.BM25_B):
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, term)
        bm_25tf = (tf * (k1 + 1) / (tf + k1 * length_norm))
        return bm_25tf

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        doc_length = 0 
        for doc in self.doc_lengths:
            doc_length += self.doc_lengths[doc]            
        return doc_length/len(self.doc_lengths)

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit=5):
        query_tokens = preprocess(query)
        scores = {}
        for doc_id in self.docmap:
            total = 0
            for token in query_tokens:
                total += self.bm25(doc_id, token)
            scores[doc_id] = total
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_items[:limit]

    def build(self):
        data = load_movies()
        movies = next(iter(data.values()), None)
        for movie in movies:
            doc_id = movie["id"]
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")
            self.docmap[doc_id] = movie
    
    def save(self):
        os.makedirs('cache', exist_ok=True)
        index_file_path = "cache/index.pkl"
        docmap_file_path = "cache/docmap.pkl"
        term_frequencies_file_path = "cache/term_frequencies.pkl"
        doc_lengths_path = "cache/doc_lengths.pkl"
        with open (index_file_path, 'wb') as file:
            dump(self.index, file)
        with open (docmap_file_path, 'wb') as file:
            dump(self.docmap, file)
        with open(term_frequencies_file_path, 'wb') as file:
            dump(self.term_frequencies, file)  
        with open(doc_lengths_path, 'wb') as file:
            dump(self.doc_lengths, file)

    def load_from_cache(self):
        index_file_path = "cache/index.pkl"
        docmap_file_path = "cache/docmap.pkl"
        term_frequencies_path = "cache/term_frequencies.pkl"
        doc_lengths_path = "cache/doc_lengths.pkl"
        with open(index_file_path, 'rb') as file:
            index = load(file)
            self.index = index
        with open(docmap_file_path, 'rb') as file:
            docmap = load(file)
            self.docmap = docmap
        with open(term_frequencies_path, 'rb') as file:
            term_frequencies = load(file)
            self.term_frequencies = term_frequencies
        with open(doc_lengths_path, 'rb') as file:
            doc_length = load(file)
            self.doc_lengths = doc_length
def preprocess(raw_string):
    lowered_string = raw_string.lower()
    stripped_string = lowered_string.translate(str.maketrans("", "", string.punctuation))
    tokens = tokenization(stripped_string)
    filtered_tokens = remove_stop_words(tokens)
    stemmed_tokens = stemmer(filtered_tokens)
    return stemmed_tokens
    
def tokenization(string):
    tokens = string.strip().split()
    while "" in tokens:
        tokens.remove("")
    return tokens

def remove_stop_words(words):
    res = []
    for word in words:
        if word not in stopwords:
            res.append(word)
    return res
    
def process_stopwords(raw_stopwords):
    result = []
    for raw_stopword in raw_stopwords:
        result.append(raw_stopword.lower().translate(str.maketrans("", "", string.punctuation)))
    return result

def stemmer(tokens):
    stemmer = PorterStemmer()
    result = []
    for token in tokens:
        result.append(stemmer.stem(token))
    return result

def tokenize_term(term):
    token = preprocess(term)
    if len(token) != 1:
        raise ValueError('Length of token not 1')
    return token[0]

def load_movies():
    with open('./data/movies.json', 'r') as file:
        data = json.load(file)
    return data

def load_stopwords():
    with open('./data/stopwords.txt', 'r') as file:
        stopwords = file.read().splitlines()
    return stopwords
