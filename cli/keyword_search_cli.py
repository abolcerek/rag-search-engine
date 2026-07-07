import argparse
import json
import string
import math
import config
from collections import Counter
from nltk.stem import PorterStemmer
from pickle import dump, load
import os

def main() -> None:
    
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


    results = []
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Avaliable commands")
    
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Inverted Index from movies")

    tf_parser = subparsers.add_parser("tf", help="Return term frequencies")
    tf_parser.add_argument("id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Inputted term")

    idf_parser = subparsers.add_parser("idf", help="Get idf value for term")
    idf_parser.add_argument("term", type=str, help="Inputted term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a given term")
    tfidf_parser.add_argument("id", type=int, help="Inputted doc id")
    tfidf_parser.add_argument("term", type=str, help="Inputted term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=config.BM_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=config.BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    
    InvIdx = InvertedIndex()
    stopwords = load_stopwords()
    stopwords = process_stopwords(stopwords)
    
    match args.command:
        case "search":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return            
            query_tokens = preprocess(args.query)
            doc_ids = []
            for query_token in query_tokens:
                ids = InvIdx.get_documents(query_token)
                doc_ids.extend(ids)
            for i, id in enumerate(doc_ids[:5]):
                movie = InvIdx.docmap[id]
                print(f'{movie['id']} {movie['title']}')
        case "build":
            InvIdx.build()
            InvIdx.save()
        case "tf":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return  
            term = tokenize_term(args.term)
            tf = InvIdx.get_tf(args.id, term)
            print(f'{args.term} {tf} {args.id}')
        case "idf":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return  
            term = tokenize_term(args.term)
            doc_ids = InvIdx.get_documents(term)
            idf = math.log((len(InvIdx.docmap) + 1) / (len(doc_ids) + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return    
            term = tokenize_term(args.term)
            tf = InvIdx.get_tf(args.id, term)
            doc_ids = InvIdx.get_documents(term)
            idf = math.log((len(InvIdx.docmap) + 1) / (len(doc_ids) + 1)) 
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tf_idf:.2f}") 
        case "bm25idf":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return  
            term = tokenize_term(args.term)
            bm25idf = InvIdx.get_bm25_idf(term)           
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return  
            term = tokenize_term(args.term)
            bm_25tf = InvIdx.get_bm25_tf(args.id, term, args.k1)          
            print(f"BM25 TF score of '{args.term}' in document '{args.id}': {bm_25tf:.2f}")

        case "bm25search":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return  
            scores = InvIdx.bm25_search(args.query)
            count = 1
            for score in scores:
                movie = InvIdx.docmap[score[0]]
                movie_title = movie['title']
                print(f'{count}. ({score[0]}) {movie_title} - Score: {score[1]:.2f}')
                count += 1
        


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()