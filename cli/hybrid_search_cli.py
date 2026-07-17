import argparse
from lib import hybrid_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Avaliable commands")

    normalize = subparser.add_parser("normalize", help="Normalize a list of inputted values")
    normalize.add_argument("values", type=float, nargs="*", help="List of inputted values")
 

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if len(args.values) == 0:
                return
            scores = hybrid_search.normalize(args.values)
            for score in scores:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()