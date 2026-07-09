import argparse
from lib import semantic_search
import json

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Avaliable commands")

    verify_parser = subparser.add_parser("verify", help="Verify embedding model")

    verify_embeddings_parser = subparser.add_parser("verify_embeddings", help="Verify embeddings")

    embedding_parser = subparser.add_parser("embed_text", help="Embed inputted text")
    embedding_parser.add_argument("text", type=str, help="Inputted text to be embedded")

    embed_query = subparser.add_parser("embed_query", help="Embed an inputted query")
    embed_query.add_argument("query", type=str, help="Inputted query to be embedded")

    search = subparser.add_parser("search", help="Semantic search command")
    search.add_argument("query", type=str, help="Inputted search query")
    search.add_argument("--limit", type=int, nargs='?', default=5, help="Optional limit parameter")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            semantic_search.embed_text(args.text)
        case "verify_embeddings":
            semantic_search.verify_embeddings()
        case "embed_query":
            semantic_search.embed_query_text(args.query)
        case "search":
            sem_search = semantic_search.SemanticSearch()
            movies = []
            with open('./data/movies.json', 'r') as file:
                data = json.load(file)
            movies.extend(data['movies'])
            sem_search.load_or_create_embeddings(movies)
            search_results = sem_search.search(args.query, args.limit)
            for i in range(len(search_results)):
                print(f"{i + 1}. {search_results[i]['title']} (score: {search_results[i]['score']:.4f}) \n{search_results[i]['description']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
