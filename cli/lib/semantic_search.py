from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}


    def generate_embeddings(self, text):
        if len(text) == 0 or len(text.strip()) == 0:
            raise ValueError("Text must not be empty")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        movies = []
        for document in self.documents:
            self.document_map[document['id']] = document
            movies.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(movies, show_progress_bar = True)
        np.save('cache/movie_embeddings', self.embeddings)

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document['id']] = document
        if os.path.exists('cache/movie_embeddings'):
            self.embeddings = np.load('cache/movie_embeddings')
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)


def verify_model():
    semsearch = SemanticSearch()
    print(f"Model loaded: {semsearch.model}")
    print(f"Max sequence length: {semsearch.model.max_seq_length}")

def embed_text(text):
    semsearch = SemanticSearch()
    embedding = semsearch.generate_embeddings(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semsearch = SemanticSearch()
    movies = []
    with open('./data/movies.json', 'r') as file:
        data = json.load(file)
    movies.extend(data['movies'])
    semsearch.load_or_create_embeddings(movies)
    print(f"Number of docs: {len(semsearch.documents)}")
    print(f"Embeddings shape: {semsearch.embeddings.shape[0]} vectors in {semsearch.embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semsearch = SemanticSearch()
    embedding = semsearch.generate_embeddings(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")