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
        self.idx.load_from_cache()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5):
        bm25_results = self._bm25_search(query, limit * 500) #doc id at score[0]
        sem_results = self.semantic_search.search_chunks(query, limit * 500) #doc id at score["id"]
        bm_25_scores = []
        sem_scores = []
        results = {}
        hybrid_scores = []
        for score in bm25_results:
            bm_25_scores.append(score[1])
        for result in sem_results:
            sem_scores.append(result["score"])
        norm_bm25_scores = normalize(bm_25_scores)
        norm_sem_scores = normalize(sem_scores)
        for i, result in enumerate(bm25_results):
            if result[0] not in results:
                document = self.idx.docmap[result[0]]
                results[result[0]] = {"bm25_score": norm_bm25_scores[i], "semantic_score": 0.0}
                results[result[0]]["document"] = document
        for i, result in enumerate(sem_results):
            if result["id"] not in results:
                results[result["id"]] = {"bm25_score": 0.0,"semantic_score": norm_sem_scores[i]}
                results[result["id"]]["document"] = self.idx.docmap[result["id"]]
            else:
                results[result["id"]]["semantic_score"] = norm_sem_scores[i]
                results[result["id"]]["document"] = self.idx.docmap[result["id"]]
        for key, value in results.items():
            hyb_score = hybrid_score(value["bm25_score"], value["semantic_score"], alpha)
            results[key]["hybrid_score"] = hyb_score
        sorted_scores = sorted(results.items(), key=lambda item: item[1]["hybrid_score"], reverse=True)
        return sorted_scores[:limit]      

    def rrf_search(self, query: str, k: int, limit: int = 10):
        bm25_results = self._bm25_search(query, limit * 500) #doc id at score[0]
        sem_results = self.semantic_search.search_chunks(query, limit * 500) #doc id at score["id"]
        results = {}
        for i, result in enumerate(bm25_results):
            if result[0] not in results:
                document = self.idx.docmap[result[0]]
                results[result[0]] = {"bm25_rank": i + 1}
                results[result[0]]["document"] = document
        for i, result in enumerate(sem_results):
            if result["id"] not in results:
                results[result["id"]] = {"semantic_rank": i + 1}
                results[result["id"]]["document"] = self.idx.docmap[result["id"]]
            else:
                results[result["id"]]["semantic_rank"] = i + 1
                results[result["id"]]["document"] = self.idx.docmap[result["id"]]
        for key, value in results.items():
            rrf_bm25 = rrf_score(value["bm25_rank"], k)
            rrf_semantic = rrf_score(value["semantic_rank"], k)
            results[key]["rrf_score"] = rrf_bm25 + rrf_semantic
        sorted_scores = sorted(results.items(), key=lambda item: item[1]["rrf_score"], reverse=True)
        return sorted_scores[:limit]


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

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)