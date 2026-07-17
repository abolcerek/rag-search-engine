from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import re

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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self):
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents): 
        self.documents = documents
        chunks = []
        chunks_metadata = []
        for i, document in enumerate(self.documents):
            self.document_map[document['id']] = document
            if len(document['description']) == 0:
                continue
            sem_chunks = semantic_chunk(document['description'], 4, 1)
            for chunk_index, chunk_text in enumerate(sem_chunks):
                chunks.append(chunk_text)
                chunks_metadata.append({"movie_idx": i,  "chunk_idx": chunk_index, "total_chunks": len(sem_chunks)})
        self.chunk_embeddings = self.model.encode(chunks)
        np.save('cache/chunk_embeddings', self.chunk_embeddings)
        self.chunk_metadata = {"chunks": chunks_metadata, "total_chunks": len(chunks)}
        with open ('cache/chunk_metadata.json', 'w') as file:
            json.dump({"chunks": chunks_metadata, "total_chunks": len(chunks)}, file, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document['id']] = document
        if os.path.exists('cache/chunk_embeddings.npy') and os.path.exists('cache/chunk_metadata.json'):
            self.chunk_embeddings = np.load('cache/chunk_embeddings.npy')
            with open('cache/chunk_metadata.json', 'rb') as file:
                chunk_metadata = json.load(file)
                self.chunk_metadata = chunk_metadata
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query, limit=10):
        embedded_query = self.generate_embeddings(query)
        chunk_scores = []
        for i, chunk in enumerate(self.chunk_embeddings):
            cos_sim = cosine_similarity(embedded_query, self.chunk_embeddings[i])
            chunks = self.chunk_metadata["chunks"]
            chunk_scores.append({"chunk_idx": chunks[i]["chunk_idx"], "movie_idx": chunks[i]["movie_idx"], "score": cos_sim})
        movies_scores = {}
        for i, chunk in enumerate(chunk_scores):
            if chunk['movie_idx'] not in movies_scores or chunk['score'] > movies_scores[chunk['movie_idx']]:
                movies_scores[chunk['movie_idx']] = chunk['score']
        sorted_scores = sorted(movies_scores.items(), key= lambda item : item[1], reverse=True)
        res = []
        for index, (key, value) in enumerate(sorted_scores):
            if index == limit:
                return res
            document = self.documents[key]
            res.append(format_search_result(document['id'], document['title'], document['description'][:100], value))
        return res 
        

def semantic_chunk(text: str, max_chunk_size: int = 5, overlap: int = 1) -> list[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]
    chunks: list[str] = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            chunk_sentence = chunk_sentence.strip()
            if chunk_sentence:
                cleaned_sentences.append(chunk_sentence)
        if not cleaned_sentences:
            i += max_chunk_size - overlap
            continue
        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)
        i += max_chunk_size - overlap
    return chunks


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


def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, 2),
        "metadata": metadata if metadata else {},
    }