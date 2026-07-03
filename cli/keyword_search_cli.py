import argparse
import json

def main() -> None:
    results = []
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Avaliable commands")
    
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    
    args = parser.parse_args()
    
    with open('./data/movies.json', 'r') as file:
        data = json.load(file)
    
    match args.command:
        case "search":
            print(f'Searching for: {args.query}')
            for key, value in data.items():
                print(f'This is the key: {key}\n')
                print(f'This is the value: {value}\n')
                
        
        case _:
            parser.print_help()
    
if __name__ == "__main__":
    main()