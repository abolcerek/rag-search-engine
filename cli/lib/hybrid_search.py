from .semantic_search import ChunkedSemanticSearch
from .keyword_search import InvertedIndex
import os


class HybridSearch():
    def __init__(self, documents):
        self.documents =  documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        
        self.idx = InvertedIndex()
        try:  
            self.idx.load_from_cache()
        except FileNotFoundError:
            self.idx.build()
            self.idx.save()
    
    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5):
        raise NotImplementedError("Weighted search is not implemented yet")
        # bm25_results = self._bm25_search(query, limit * 500)
        # sem_results = self.semantic_search.search_chunks(query, limit * 500)
        # bm_25_scores = []
        # sem_scores = []
        # for score in bm25_results:
        #     bm_25_scores.append(score[1])
        # for result in sem_results:
        #     sem_scores.append(result["score"])
        # norm_bm25_scores = normalize(bm_25_scores)
        # norm_sem_scores = normalize(sem_scores)
        # results = []
        

    def rrf_search(self, query: str, k: int, limit: int = 10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize(values):
    res = []
    min_val = min(values)
    max_val = max(values)
    denominator = (max_val - min_val)
    for val in values:
        if denominator == 0:
            res.append(1.0)
            continue
        numerator = (val - min_val)
        score = numerator / denominator 
        res.append(score)
    return res
