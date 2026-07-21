import argparse
from lib import hybrid_search
import json
import os
from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Avaliable commands")

    normalize = subparser.add_parser("normalize", help="Normalize a list of inputted values")
    normalize.add_argument("values", type=float, nargs="*", help="List of inputted values")

    weighted_search = subparser.add_parser("weighted-search", help="Weighted search")
    weighted_search.add_argument("text", type=str, help="Inputted query")
    weighted_search.add_argument("--alpha", type=float, nargs="?", default= 0.5, help="Optional alpha parameter")
    weighted_search.add_argument("--limit", type=int, nargs="?", default=25, help="Optional limit parameter")

    rrf_search = subparser.add_parser("rrf-search", help="Reciprocal Rank Fusion search")
    rrf_search.add_argument("text", type=str, help="Inputted query used for search")
    rrf_search.add_argument("-k", type=int, nargs="?", default=60, help="Optional k parameter")
    rrf_search.add_argument("--limit", type=int, nargs="?", default=5, help="Optional limit paramter")
    rrf_search.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")

    args = parser.parse_args()
    description_limit = 100
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

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
        case "rrf-search":
            movies = []
            with open('./data/movies.json', 'r') as file:
                data = json.load(file)
            movies.extend(data['movies'])
            HybridSearch = hybrid_search.HybridSearch(movies)
            if args.enhance:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key
                )
                if args.enhance == "spell":
                    response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": f"Fix any spelling errors in the user-provided movie search query below. Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words. Preserve punctuation and capitalization unless a change is required for a typo fix. If there are no spelling errors, or if you're unsure, output the original query unchanged. Output only the final query text, nothing else. User query: '{args.text}'",
                            }
                        ],
                        model="openrouter/free"
                    )
                    query = response.choices[0].message.content
                if args.enhance == "rewrite":
                    message = f"""Rewrite the user-provided movie search query below to be more specific and searchable.
                                Consider:
                                - Common movie knowledge (famous actors, popular films)
                                - Genre conventions (horror = scary, animation = cartoon)
                                - Keep the rewritten query concise (under 10 words)
                                - It should be a Google-style search query, specific enough to yield relevant results
                                - Don't use boolean logic

                                Examples:
                                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                                If you cannot improve the query, output the original unchanged.
                                Output only the rewritten query text, nothing else.

                                User query: "{args.text}"
                                """
                    response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": message,
                            }
                        ],
                        model="openrouter/free"
                    )
                    query = response.choices[0].message.content
                if args.enhance == "expand":
                    message = f"""Expand the user-provided movie search query below with related terms.

                    Add synonyms and related concepts that might appear in movie descriptions.
                    Keep expansions relevant and focused.
                    Output only the additional terms; they will be appended to the original query.

                    Examples:
                    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                    - "action movie with bear" -> "action thriller bear chase fight adventure"
                    - "comedy with bear" -> "comedy funny bear humor lighthearted"

                    User query: "{args.text}"
                    """
                    response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": message,
                            }
                        ],
                        model="openrouter/free"
                    )
                    query = response.choices[0].message.content
                print(f"Enhanced query ({args.enhance}): '{args.text}' -> '{query}'\n")
            else:
                query = args.text
            results = HybridSearch.rrf_search(query, args.k, args.limit)
            for i, (doc_id, result) in enumerate(results):
                print(f'{i + 1}. {result['document']['title']}\n RRF Score: {result['rrf_score']:.3f}\n BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n {result['document']['description'][:description_limit]}...\n')
        
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()