import argparse
from lib import semantic_search
import json
import re

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

    chunk = subparser.add_parser("chunk", help="Chunk the inputted text")
    chunk.add_argument("text", type=str, help="Inputted text to be chunked")
    chunk.add_argument("--chunk-size", type=int, nargs="?", default=200, help="Optional chunk size parameter")
    chunk.add_argument("--overlap", type=int, help="Parameter for chunk overlap")

    semantic_chunk = subparser.add_parser("semantic_chunk", help="Semantically chunk the inputted text")
    semantic_chunk.add_argument("text", type=str, help="Inputted text to be chunked")
    semantic_chunk.add_argument("--max-chunk-size", type=int, nargs="?", default=4, help="Optional max chunk size parameter")
    semantic_chunk.add_argument("--overlap", type=int, nargs="?", default=0, help="Optional overlap parameter")

    embed_chunk = subparser.add_parser("embed_chunks", help="Embed movie documents into chunks")

    search_chunk = subparser.add_parser("search_chunked", help="Search the chunked documents")
    search_chunk.add_argument("text", type=str, help="Inputted text to be searched")
    search_chunk.add_argument("--limit", type=int, nargs="?", default=5, help="Optional limit parameter for search")



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
        case "chunk":
            character_length = len(args.text)
            chunks = args.text.split()
            print(f"Chunking {character_length} characters")
            counter = 1
            i = 0
            while i < character_length:
                list_of_chunks = chunks[i: i + args.chunk_size]
                if len(list_of_chunks) == 0:
                    break
                print(f"{counter}. {" ".join(list_of_chunks)}")
                counter += 1
                i += args.chunk_size - args.overlap

        case "semantic_chunk":
            chunks = semantic_search.semantic_chunk(args.text)
            if len(chunks) == 0:
                return
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")
        case "embed_chunks":
            movies = []
            with open('./data/movies.json', 'r') as file:
                data = json.load(file)
            movies.extend(data['movies'])
            chunk_sem_search = semantic_search.ChunkedSemanticSearch()
            embeddings = chunk_sem_search.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            movies = []
            with open('./data/movies.json', 'r') as file:
                data = json.load(file)
            movies.extend(data['movies'])
            chunk_sem_search = semantic_search.ChunkedSemanticSearch()
            chunk_sem_search.load_or_create_chunk_embeddings(movies)
            results = chunk_sem_search.search_chunks(args.text, args.limit)
            for i, result in enumerate(results):
                print(f"\n{i + 1}. {result["title"]} (score: {result["score"]:.4f})")
                print(f"    {result["document"]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
