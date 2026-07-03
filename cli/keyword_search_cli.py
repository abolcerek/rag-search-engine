import argparse
import json
import string
from nltk.stem import PorterStemmer

def main() -> None:
    
    def preprocess(raw_string):
        lowered_string = raw_string.lower()
        stripped_string = lowered_string.translate(str.maketrans("", "", string.punctuation))
        tokens = tokenization(stripped_string)
        filtered_tokens = remove_stop_words(tokens)
        stemmed_tokens = stemmer(filtered_tokens)
        return stemmed_tokens
        
    def tokenization(string):
        tokens = string.strip().split()
        while "" in tokens:
            tokens.remove("")
        return tokens

    def remove_stop_words(words):
        res = []
        for word in words:
            if word not in stopwords:
                res.append(word)
        return res
        
    def process_stopwords(raw_stopwords):
        result = []
        for raw_stopword in raw_stopwords:
            result.append(raw_stopword.lower().translate(str.maketrans("", "", string.punctuation)))
        return result

    def stemmer(tokens):
        stemmer = PorterStemmer()
        result = []
        for token in tokens:
            result.append(stemmer.stem(token))
        return result

    results = []
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Avaliable commands")
    
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    
    args = parser.parse_args()
    
    with open('./data/movies.json', 'r') as file:
        data = json.load(file)

    with open('./data/stopwords.txt', 'r') as file:
        stopwords = file.read().splitlines()
    
    match args.command:
        case "search":
            stopwords = process_stopwords(stopwords)
            query_tokens = preprocess(args.query)
            movies = next(iter(data.values()), None)
            result = 1
            for item in movies:
                if result == 6:
                    break
                title_tokens = preprocess(item["title"])
                matched = False
                for query_token in query_tokens:
                    if matched:
                        break
                    for title_token in title_tokens:                        
                        if query_token in title_token:
                            matched = True
                            break
                if matched:
                    print(f'{result}. {item["title"]} {result}')
                    result += 1
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()