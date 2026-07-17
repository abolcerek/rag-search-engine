import argparse
import json
import string
import math
import config
from collections import Counter
from nltk.stem import PorterStemmer
from pickle import dump, load
import os
from lib import keyword_search

def main() -> None:

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
    
    InvIdx = keyword_search.InvertedIndex()
    stopwords = keyword_search.load_stopwords()
    stopwords = keyword_search.process_stopwords(stopwords)
    
    match args.command:
        case "search":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return            
            query_tokens = keyword_search.preprocess(args.query)
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
            term = keyword_search.tokenize_term(args.term)
            tf = InvIdx.get_tf(args.id, term)
            print(f'{args.term} {tf} {args.id}')
        case "idf":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return  
            term = keyword_search.tokenize_term(args.term)
            doc_ids = InvIdx.get_documents(term)
            idf = math.log((len(InvIdx.docmap) + 1) / (len(doc_ids) + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return    
            term = keyword_search.tokenize_term(args.term)
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
            term = keyword_search.tokenize_term(args.term)
            bm25idf = InvIdx.get_bm25_idf(term)           
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            try:  
                InvIdx.load_from_cache()
            except FileNotFoundError:
                print('Inverted index file not found within the cache')
                return  
            term = keyword_search.tokenize_term(args.term)
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