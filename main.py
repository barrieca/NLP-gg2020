import json
import spacy
import time
import imdb
import Levenshtein
import pandas as pd

def find_truncated_candidates(df, search_type, min=1):
    '''
    Finds a statistically truncated list of candidates that are in IMDb.
    :param df: DataFrame of sorted nouns
    :param search_type: 'name' for person(s) or 'title' for movie(s)
    :param min: The number of candidates (default is 1).
    :return: The list of candidates.
    '''
    return statistical_truncation(find_imdb_objects(df, search_type, min), 0.7, min)

def find_imdb_objects(df, search_type, n=1):
    '''
    Returns list of tuples of n noun chunks that were successfully found on IMDb, and their frequency.
    :param df: DataFrame of sorted nouns
    :param search_type: 'name' for person(s) or 'title' for movie(s)
    :param n: The number of candidates (default is 1).
    :return:
    Written by Cameron.
    '''
    imdb_obj = imdb.IMDb()
    search_function = (imdb_obj.search_person if search_type == 'name' else imdb_obj.search_movie)
    results = []
    for i, row in df.iterrows():
        noun = row['text']
        result = search_function(noun)
        if len(result) > 0 and result[0][search_type] == noun:
            results.append((noun,row['freq']))
        if len(results) >= n:
            break
    return results

def aggregate_and_sort_df(df):
    df['freq'] = df.groupby('word')['word'].transform('count')
    return df.drop_duplicates().sort_values(by='freq', ascending=False)

def find_tweets_about_host(data, word_list = []):
    nlp = spacy.load('en_core_web_sm')
    for d in data:
        for i in d.split(' '):
            if i[:4]=='host':
                var = nlp(d)
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

def filter_tweets(df_tweets, regex_string):
    '''
    Filters the dataframe of tweet text to only include those that match the given regex_string.
    :param df_tweets: The dataframe containing the tweets' text
    :param regex_string: The regular expression string to apply to each tweet (i.e. movie|film|picture)
    :return:
    Example: noun_df = filter_tweets(pd.DataFrame(tweet_list, columns=['text']), 'movie|film|picture')
    '''
    return df_tweets[df_tweets.text.str.contains(regex_string, regex=True)]


def create_noun_chunks(df_tweet):
    '''
    Produces the noun chunks in the text column of the tweets.
    :param df_tweet: A dataframe of tweets with the column 'text'.
    :return: Returns a dataframe of the noun chunks in the tweet text.
    '''
    # Instantiate spacy
    nlp = spacy.load('en_core_web_sm')

    # Apply the noun chunking to the remaining text
    test_func = lambda x: [*nlp(x).noun_chunks]
    # df_tweet = df_tweet.apply(lambda x: [*nlp(x).noun_chunks], axis=1)
    print(df_tweet.values)
    df_noun_chunks = pd.DataFrame()
    df_noun_chunks['text'] = test_func(df_tweet['text'].values)
    return df_tweet


def get_noun_frequencies(df_nouns):
    '''
    Returns sorted data frame of unique, candidate nouns.
    :param df_nouns:
    :return:
    '''
    df_nouns['freq'] = df_nouns.groupby('text')['text'].transform('count')
    return df_nouns.drop_duplicates().sort_values(by='freq', ascending=False)

def statistical_truncation(list_candidates, threshold_percent, min = 0):
    '''

    :param list_candidates:
    :param threshold_percent:
    :param min:
    :return: List of answers.
    Example Usage:
    tup_list = [('John',500),('Jane',450),('Jim',400),('Jake',399),('Jesse',300)]
    statistical_truncation(tup_list,0.8) = ['John', 'Jane', 'Jim']
    '''
    print(list_candidates)
    top_frequency = list_candidates[0][1]
    result_list = []
    for candidate in list_candidates:
        if candidate[1] < top_frequency * threshold_percent and len(result_list) >= min:
            break
        else:
            result_list.append(candidate[0])
    return result_list

def split_data_by_time(json_data, start_time):
    '''
    Builds two dataframes of the tweets with tweets before and after the starting time of the GG.
    :param json_data: The JSON data
    :param start_time: The start time of the Golden Globes in UTC datetime format.
    :return: Two dataframes of tweets (pre-show and non-pre-show) with desirable qualities.
    '''
    df_tweets = pd.DataFrame(json_data)

    df_tweets_after_start = df_tweets[pd.to_datetime(df_tweets['created_at']) >= start_time]
    df_tweets_before_start = df_tweets[pd.to_datetime(df_tweets['created_at']) < start_time]

    return df_tweets_before_start[['text']], df_tweets_after_start[['text']]


    # negative_words = ['didn\'t', 'not', 'should'] # we can change this list as we see fit
    # preshow_words = ['hope','predict','opinion','want','belie'] # again, we can change this as we see fit, but I think a time analysis would work better
    # pre_show = []
    # non_pre_show = []
    #
    # nlp = spacy.load('en_core_web_sm')
    # for d in json_data:
    #     add = True
    #     preshow = False
    #     for i in d['text'].split(' '):
    #         if any([i.startswith(word) for word in negative_words]):
    #             add = False
    #             break
    #         elif any([i.startswith(word) for word in preshow_words]):
    #             preshow = True
    #     if add:
    #         if preshow:
    #             pre_show.append(d['text'])
    #         else:
    #             non_pre_show.append(d['text'])
    #     #if not any([(any([i.startswith(word) for word in negative_words])) for i in d['text'].split(' ')]):
    #     #    non_pre_show.append(d['text'])
    # return (pre_show, non_pre_show)

def get_nominees(df_tweets):
    '''
    Determines the nominees for each award based on the given list of tweets.
    :param pre_processed_tweet_list: A list of tweets that have been pre-filtered.
    :return: Dictionary containing 27 keys, with list as its value
    '''
    num_possible_nominees = 5
    award_nominees = {}

    # For each award category
    for category in award_names:
        #try:
        #for i in range(0,1):
         #   category = award_names[i]
        # Filter tweets by subject string
        # nominee_tweets = find_tweets_about(pre_processed_tweet_list, 'win')
        df_nominee_tweets = filter_tweets(df_tweets, "win")
        print("filtered nominee tweets")
        # Get the nouns chunks in the remaining tweets
        df_noun_chunks = create_noun_chunks(df_nominee_tweets)
        print("found noun chunks")
        # Aggregate and sort the noun chunks
        df_sorted_nouns = get_noun_frequencies(df_noun_chunks)
        print("found noun frequencies")
        # Produce the correct number of noun chunks that also exist on IMDb
        imdb_candidates = find_imdb_objects(df_sorted_nouns, 'title', 10)

        # trunc_candidates = statistical_truncation(imdb_candidates, 0.7, 5)

        award_nominees[category] = imdb_candidates
        print("found the award nominees")
      #  except:
      #      continue

    return award_nominees

def fuzzy_match(s1, s2, threshold):
    '''
    Uses Levenshtein distance to determine how well two strings match.
    :param s1: String One
    :param s2: String Two
    :param threshold:
    :return: Returns boolean based on whether the two strings match within some threshold of the Levenshtein distance.
    '''
    dist = Levenshtein.distance(s1, s2)
    base_len = len(s1)
    return (dist <= round(base_len * threshold))


award_names = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']

def main():

    # Read in JSON data
    data = [json.loads(line) for line in open('gg2020.json','r',encoding='utf-8')]

    # Split data into two dataframes: pre-show and after show starts
    pre_data, data = split_data_by_time(data, pd.to_datetime('2020-01-06T01:00:00'))

    print(get_nominees(data))

    # word_list = find_tweets_about_host(data)
    # word_df = pd.DataFrame(word_list, columns=['word'])
    # host = find_imdb_objects(aggregate_and_sort_df(word_df), 'name')[0][0]
    # print(host)



t = time.time()
main()
print(time.time()-t)

# award_names = ["Best Motion Picture – Drama",
#                "Best Motion Picture – Musical or Comedy",
#                "Best Motion Picture – Foreign Language",
#                "Best Motion Picture – Animated",
#                "Best Director – Motion Picture",
#                "Best Actor – Motion Picture Drama",
#                "Best Actor – Motion Picture Musical or Comedy",
#                "Best Actress – Motion Picture Drama",
#                "Best Actress – Motion Picture Musical or Comedy",
#                "Best Supporting Actor – Motion Picture",
#                "Best Supporting Actress – Motion Picture",
#                "Best Screenplay – Motion Picture",
#                "Best Original Score – Motion Picture",
#                "Best Original Song – Motion Picture",
#                "Cecil B. DeMille Award for Lifetime Achievement in Motion Pictures",
#                "Best Television Series – Drama",
#                "Best Television Series – Musical or Comedy",
#                "Best Miniseries or Television Film",
#                "Best Actor – Television Series Drama",
#                "Best Actor – Television Series Musical or Comedy",
#                "Best Actor – Miniseries or Television Film",
#                "Best Actress – Television Series Drama",
#                "Best Actress – Television Series Musical or Comedy",
#                "Best Actress – Miniseries or Television Film",
#                "Best Supporting Actor – Series,  Miniseries or Television Film",
#                "Best Supporting Actress – Series, Miniseries or Television Film",
#                "Carol Burnett Award for Lifetime Achievement in Television"]