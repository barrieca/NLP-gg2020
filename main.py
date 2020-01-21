import json
import nltk
import Levenshtein

def fuzzy_match(base_str, candidate_str, threshold):
    dist = Levenshtein.distance(base_str, candidate_str)
    base_len = len(base_str)
    return (dist <= round(base_len * threshold))

def main():
    # download nltk corpa
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    # load tweets
    with  open('gg2020.json','rb') as tweetfile:
        tweets = json.load(tweetfile)['tweets']
    current_tweet = tweets[0]
    words = nltk.word_tokenize(current_tweet['text'])
    tags = nltk.pos_tag(words)
    print(tags)

    # Just testing functions
    print(Levenshtein.editops('Priyanka','Pryanka'))
    print(fuzzy_match('Priyanka', 'Pryanka', 0.1))
    print(fuzzy_match('Priyanka', 'Pryank', 0.1))

if __name__ == "__main__":
    main()
