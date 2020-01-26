import json
import spacy
import time
import imdb
import pandas as pd

def find_imdb_person(df):
    imdb_obj = imdb.IMDb()
    noun = df.iloc[0]['word']
    result = imdb_obj.search_person(noun)
    if len(result) == 0 or result[0]['name'] != noun:
        return find_imdb_person(df.iloc[1:,])
    else:
        return noun

def aggregate_and_sort_df(df):
    df['freq'] = df.groupby('word')['word'].transform('count')
    return df.drop_duplicates().sort_values(by='freq', ascending=False)

def find_tweets_about_host(data, word_list = []):
    nlp = spacy.load('en_core_web_sm')
    for d in data:
        for i in d['text'].split(' '):
            if i[:4]=='host':
                var = nlp(d['text'])
                for word in [*var.noun_chunks]:
                    word = word.text.strip('•').strip(' ')
                    word_list.append(word)
                break
    return word_list

def find_tweets_about(tweet_list, subject):
    '''
    Searches each tweet in the list for the subject.
    :param tweet_list: The list of tweets to search through.
    :param subject: String denoting the subject to look for.
    :return: Returns new list of relevant tweets containing the subject.
    Written by Marko.
    '''
    candidate_tweets = []
    for tweet in tweet_list:
        if subject in tweet:
            candidate_tweets.append(tweet)
    return candidate_tweets

def create_noun_chunks(tweet_list, award_name):
    '''
    Returns dataframe of noun chunks.
    :param tweet_list:
    :param award_name:
    :return:
    Written by Alex.
    '''
    print("Not implemented yet")


def get_noun_frequencies(df_nouns):
    '''
    Returns sorted data frame of unique, candidate nouns.
    :param df_nouns:
    :return:
    Written by Alex.
    '''
    print("Not implemented yet")

def filter_with_imdb(df_sorted_nouns, n):
    '''
    Returns list of tuples of n noun chunks that were successfully found on IMDb, and their frequency.
    :param df_sorted_nouns:
    :param n: The number of candidates.
    :return:
    Written by Cameron.
    '''
    print("Not implemented yet")

def statistical_truncation(list_candidates, threshold_percent, min):
    '''

    :param list_candidates:
    :param threshold_percent:
    :param min:
    :return: List of answers.
    Written by Cameron.
    '''
    print("Not implemented yet")

def pre_process_data(json_data):
    '''

    :param json_data:
    :return: Two lists of tweets (pre-show and non-pre-show) with desirable qualities.
    Written by Cameron.
    '''
    print("Not implemented yet")

def get_nominees(pre_processed_tweet_list):
    '''
    Determines the nominees for each award based on the given list of tweets.
    :param pre_processed_tweet_list: A list of tweets that have been pre-filtered.
    :return: Dictionary containing 27 keys, with list as its value
    Written by Marko.
    '''

    num_possible_nominees = 5
    award_nominees = {}

    # For each award category
    for category in award_names:
        # Filter tweets by subject string
        nominee_tweets = find_tweets_about(pre_processed_tweet_list, 'nomin')

        # Get the nouns chunks in the remaining tweets
        df_noun_chunks = create_noun_chunks(nominee_tweets, category)

        # Aggregate and sort the noun chunks
        df_sorted_nouns = aggregate_and_sort_df(df_noun_chunks)

        # Produce the correct number of noun chunks that also exist on IMDb
        candidate_list = filter_with_imdb(df_sorted_nouns, num_possible_nominees)

        # Perform statistical truncation to get plausible results amd add to map for the current award
        award_nominees[category] = statistical_truncation(candidate_list, 0.1, num_possible_nominees)

    return award_nominees

def fuzzy_match(s1, s2, threshold):
    '''

    :param s1:
    :param s2:
    :param threshold:
    :return:
    Written by Cameron.
    '''
    print("Not implemented yet")

def find_imdb_movie(df_row):
    '''
    Searches IMDb to see if the movie title in df_row exists.
    :param df_row: The movie for which to search IMDb.
    :return: The proper name of the movie.
    Written by Marko.
    '''
    imdb_obj = imdb.IMDb()
    noun = df_row['word']
    result = imdb_obj.search_movie(noun)
    if len(result) == 0 or result[0]['title'] != noun:
        return None
    else:
        return noun

award_names = []

def main():
    data = [json.loads(line) for line in open('gg2020.json','r',encoding='utf-8')]
    # word_list = find_tweets_about_host(data)
    # word_df = pd.DataFrame(word_list, columns=['word'])
    # host = find_imdb_person(aggregate_and_sort_df(word_df))
    # print(host)
    tweet_list = []
    for d in data:
        tweet_list.append(d['text'])
    candidate_tweets = find_tweets_about(tweet_list, 'nomin')
    for i in range(0, 20):
        print(candidate_tweets[i])
        print()


t = time.time()
main()
print(time.time()-t)


