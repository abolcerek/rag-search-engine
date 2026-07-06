import argparse
import json
import string
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

    
        def __add_document(self, doc_id, text):
            tokenized_text = preprocess(text)
            self.term_frequencies[doc_id] = Counter()
            for token in tokenized_text:
                if token in self.index:
                    self.index[token].add(doc_id)
                else:
                    self.index[token] = {doc_id}
                self.term_frequencies[doc_id][token] += 1

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
            with open (index_file_path, 'wb') as file:
                dump(self.index, file)
            with open (docmap_file_path, 'wb') as file:
                dump(self.docmap, file)
            with open(term_frequencies_file_path, 'wb') as file:
                dump(self.term_frequencies, file)  

        def load_from_cache(self):
            index_file_path = "cache/index.pkl"
            docmap_file_path = "cache/docmap.pkl"
            term_frequencies_path = "cache/term_frequencies.pkl"
            with open(index_file_path, 'rb') as file:
                index = load(file)
                self.index = index
            with open(docmap_file_path, 'rb') as file:
                docmap = load(file)
                self.docmap = docmap
            with open(term_frequencies_path, 'rb') as file:
                term_frequencies = load(file)
                self.term_frequencies = term_frequencies

    results = []
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Avaliable commands")
    
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Inverted Index from movies")

    tf_parser = subparsers.add_parser("tf", help="Return term frequencies")
    tf_parser.add_argument("id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Inputted term")
    
    args = parser.parse_args()
    
    InvIdx = InvertedIndex()
    stopwords = load_stopwords()
    
    match args.command:
        case "search":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return            
            stopwords = process_stopwords(stopwords)
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
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()