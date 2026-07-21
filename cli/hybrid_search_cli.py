import argparse
from lib import hybrid_search
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Avaliable commands")

    normalize = subparser.add_parser("normalize", help="Normalize a list of inputted values")
    normalize.add_argument("values", type=float, nargs="*", help="List of inputted values")

    weighted_search = subparser.add_parser("weighted-search", help="Weighted search")
    weighted_search.add_argument("text", type=str, help="Inputted query")
    weighted_search.add_argument("--alpha", type=float, nargs="?", default= 0.5, help="Optional alpha parameter")
    weighted_search.add_argument("--limit", type=int, nargs="?", default=25, help="Optional limit parameter")
 

    args = parser.parse_args()
    description_limit = 100

    match args.command:
        case "normalize":
            if len(args.values) == 0:
                return
            scores = hybrid_search.normalize(args.values)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            movies = []
            with open('./data/movies.json', 'r') as file:
                data = json.load(file)
            movies.extend(data['movies'])
            HybridSearch = hybrid_search.HybridSearch(movies)
            results = HybridSearch.weighted_search(args.text, args.alpha, args.limit)
            for i, (doc_id, result) in enumerate(results):
                print(f'{i + 1}. {result['document']['title']}\n Hybrid Score: {result['hybrid_score']:.3f}\n BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}\n {result['document']['description'][:description_limit]}...\n')
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()