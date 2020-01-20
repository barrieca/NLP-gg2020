import json
import nltk
import Levenshtein

def fuzzy_match(base_str, candidate_str, threshold):
    dist = Levenshtein.distance(base_str, candidate_str)
    base_len = len(base_str)
    return (dist <= round(base_len * threshold))

def main():
    # load tweets
    with  open('gg2020.json','rb') as tweetfile:
        temp = json.load(tweetfile)


if __name__ == "__main__":
    main()
