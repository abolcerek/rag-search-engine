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

    def search(self, query, limit):
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        embedded_query = self.generate_embeddings(query)
        list_of_cosine_similarities = []
        for i in range(len(self.embeddings)):
            cos_sim = cosine_similarity(embedded_query, self.embeddings[i])
            list_of_cosine_similarities.append((cos_sim, self.documents[i]))
        sorted_similarities = sorted(list_of_cosine_similarities, key=lambda x: x[0], reverse=True)
        res = []
        for i in range(limit):
            score = sorted_similarities[i][0]
            movie = sorted_similarities[i][1]
            dictionary = {"score": score, "title": movie['title'], "description": movie['description']}
            res.append(dictionary)
        return res 


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

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
