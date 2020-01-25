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
    Returns new list of relevant tweets containing subject.
    :param tweet_list:
    :param subject:
    :return:
    Written by Marko.
    '''
    print("Not implemented yet")

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

    :param pre_processed_tweet_list:
    :return: Dictionary containing 27 keys, with list as its value
    Written by Marko.
    '''
    print("Not implemented yet")

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

    :param df_row:
    :return:
    Written by Marko.
    '''
    print("Not implemented yet")

award_names = []

def main():
    data = [json.loads(line) for line in open('gg2020.json','r',encoding='utf-8')]
    word_list = find_tweets_about_host(data)
    word_df = pd.DataFrame(word_list, columns=['word'])
    host = find_imdb_person(aggregate_and_sort_df(word_df))
    print(host)


t = time.time()
main()
print(time.time()-t)


