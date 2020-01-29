import json
import spacy
import time
import imdb
import itertools
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

def is_valid_movie_year(test_year, award_year):
    return test_year != '????' and test_year >= award_year-2 and test_year < award_year

def is_valid_series_year(test_year, award_year):
    return test_year != '????' and test_year >= award_year-15 and test_year < award_year

def find_imdb_objects(df, search_type, year=0, n=1, is_movie=False, fuzzy_threshold=0.25):
    '''
    Returns list of tuples of n noun chunks that were successfully found on IMDb, and their frequency.
    :param df: DataFrame of sorted nouns
    :param search_type: 'name' for person(s) or 'title' for movie(s)
    :param n: The number of candidates (default is 1).
    :return:
    Possible criteria: imdb_obj.get_movie(object.movieID)['rating'] > 7.0
    '''
    imdb_obj = imdb.IMDb()
    search_function = (imdb_obj.search_person if search_type == 'name' else imdb_obj.search_movie)
    results = []
    for i, row in df.iterrows():
        noun = row['text']
        result = search_function(noun)
        if search_type == 'title':
            if is_movie:
                imdb_candidates = [object['long imdb title'][:-7] for object in result if object['kind'] == 'movie' and
                                   'year' in object and
                                   is_valid_movie_year(object['year'], year)]
                if len(imdb_candidates) > 0:
                    for candidate in imdb_candidates:
                        if fuzzy_match(candidate.lower(), noun, fuzzy_threshold):
                            results.append((candidate,row['freq']))
            else:
                imdb_candidates = [object['title'] for object in result if (object['kind'] == 'tv series' or object['kind'] == 'tv mini series' or object['kind'] == 'tv movie') and
                                   'year' in object and
                                   is_valid_series_year(object['year'], year)]
                if len(imdb_candidates) > 0:
                    for candidate in imdb_candidates:
                        if fuzzy_match(candidate.lower(), noun, fuzzy_threshold):
                            results.append((candidate, row['freq']))
        else:
            if len(result) > 0 and fuzzy_match(result[0][search_type], noun, fuzzy_threshold):
                results.append((result[0][search_type], row['freq']))
        if len(results) >= n:
            break
    return results

def get_hosts(df_tweets):
    '''
    Filters tweets containing 'host' to find, aggregates, and sorts names in the imdb database.
    :param df_tweets:
    :return:
    '''
    df_filtered_tweets = filter_tweets(df_tweets, 'host')
    print(df_filtered_tweets)
    df_filtered_tweets = get_noun_frequencies(create_noun_chunks(df_filtered_tweets))
    print(df_filtered_tweets)
    return find_truncated_candidates(df_filtered_tweets, 'name')

def filter_tweets(df_tweets, regex_string):
    '''
    Filters the dataframe of tweet text to only include those that match the given regex_string.
    :param df_tweets: The dataframe containing the tweets' text
    :param regex_string: The regular expression string to apply to each tweet (i.e. movie|film|picture)
    :return:
    Example: noun_df = filter_tweets(pd.DataFrame(tweet_list, columns=['text']), 'movie|film|picture')
    '''
    return df_tweets[df_tweets.text.str.contains(regex_string, regex=True)]

def filter_by_category(df_tweets, award_category):
    '''
    Filtering the tweets based on the award category.
    :param df_tweets: A dataframe of tweets with the column 'text'.
    :param award_category: A string representing the award category.
    :return: A filtered dataframe of tweets.
    '''

    df_filtered_tweets = pd.DataFrame(df_tweets)

    if 'picture' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'picture|movie|film')
    if 'television' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'television|series|tv')
    if 'actor' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'actor|he|him|his|[^fe]male|[^wo]man')
    if 'actress' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'actress|she|her|female|woman')
    if 'television' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'television|series|tv')
    if 'drama' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'drama')
    if 'musical' in award_category or 'comedy' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'musical|comedy|music|comed')
    if 'support' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'support')
    if 'director' in award_category:
        df_filtered_tweets= filter_tweets(df_filtered_tweets, 'direct')

    return df_filtered_tweets

def create_noun_chunks(df_tweet):
    '''
    Produces the noun chunks in the text column of the tweets.
    :param df_tweet: A dataframe of tweets with the column 'text'.
    :return: Returns a dataframe of the noun chunks in the tweet text.
    '''
    # Instantiate spacy
    nlp = spacy.load('en_core_web_sm')

    # Apply the noun chunking to the remaining text
    def find_noun_chunks(tweet_text):
        return [chunk.text.lower() for chunk in [*nlp(tweet_text).ents] if chunk.text.lower() not in noun_chunk_stop_words]

    #create a list of tweets
    array_of_tweets_text = df_tweet['text'].values.flatten()

    #create noun chunks for each individual tweet, chain the noun chunks together, and create a dataframe
    #of the noun chunks
    noun_chunks = list(itertools.chain(*[find_noun_chunks(tweet) for tweet in array_of_tweets_text]))
    df_noun_chunks = pd.DataFrame(noun_chunks, columns=['text'])

    return df_noun_chunks


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

def get_awards(df_tweets):
    return []

def get_nominees(df_tweets):
    return dict([(name, []) for name in award_names])

def get_presenters(df_tweets):
    return dict([(name, []) for name in award_names])

def get_winner(df_tweets):
    '''
    Determines the winner for each award based on the given list of tweets.
    :param pre_processed_tweet_list: A list of tweets that have been pre-filtered.
    :return: Dictionary containing 27 keys, with list as its value
    '''
    num_possible_winner = 1
    awards_year = 2020
    award_nominees = {}

    # For each award category
    for category in award_names:
        t = time.time()
        # Filter tweets by subject string
        df_nominee_tweets = filter_tweets(df_tweets, 'win|won|goes to|congratulations|congrats|congratz')

        # Filter based on the award category
        df_nominee_tweets = filter_by_category(df_nominee_tweets, category)

        print("filtered nominee tweets | " + str(df_nominee_tweets.size))

        # Get the nouns chunks in the remaining tweets
        df_noun_chunks = create_noun_chunks(df_nominee_tweets)
        print("found noun chunks")

        # Aggregate and sort the noun chunks
        df_sorted_nouns = get_noun_frequencies(df_noun_chunks)
        print("found noun frequencies")

        # Produce the correct number of noun chunks that also exist on IMDb

        imdb_candidates = find_imdb_objects(df_sorted_nouns, entity_type_to_imdb_type[award_entity_type[category]], awards_year, num_possible_winner, award_entity_type[category] == 'movie')
        print("found imdb candidates")

        # Store winner
        award_nominees[category] = imdb_candidates[0][0]
        print("found the award winner")
        print(t-time.time())

    return award_nominees

def fuzzy_match(s1, s2, threshold=0.25):
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

def entity_typer(award_name):
    '''

    :param award_name:
    :return:
    '''
    if 'actor' in award_name or 'actress' in award_name or 'director' in award_name or 'carol' in award_name or 'cecil' in award_name:
        return (award_name, 'person')
    elif 'song' in award_name:
        return (award_name, 'song')
    elif 'series' in award_name:
        return (award_name, 'tv')
    else:
        return (award_name, 'movie')

award_names = [
               'best motion picture - drama',
               'best motion picture - musical or comedy',
               'best performance by an actress in a motion picture - drama',
               'best performance by an actor in a motion picture - drama',
               'best performance by an actress in a motion picture - musical or comedy',
               'best performance by an actor in a motion picture - musical or comedy',
               'best performance by an actress in a supporting role in any motion picture',
               'best performance by an actor in a supporting role in any motion picture',
               'best director - motion picture',
               'best screenplay - motion picture',
               'best motion picture - animated',
               'best motion picture - foreign language',
               'best original score - motion picture',
               'best original song - motion picture',
               'best television series - drama',
               'best television series - musical or comedy',
               'best television limited series or motion picture made for television',
               'best performance by an actress in a limited series or a motion picture made for television',
               'best performance by an actor in a limited series or a motion picture made for television',
               'best performance by an actress in a television series - drama',
               'best performance by an actor in a television series - drama',
               'best performance by an actress in a television series - musical or comedy',
               'best performance by an actor in a television series - musical or comedy',
               'best performance by an actress in a supporting role in a series, limited series or motion picture made for television',
               'best performance by an actor in a supporting role in a series, limited series or motion picture made for television',
               'cecil b. demille award']
award_entity_type = dict(map(entity_typer, award_names))
entity_type_to_imdb_type = {'person': 'name', 'tv': 'title', 'movie': 'title'}

noun_chunk_stop_words = {'i', 'you', 'golden globe', 'golden globes', 'goldenglobes', 'congratulations', '#', 'the golden globes', 'a golden globe', 'the golden globe', 'he', 'she', 'me', 'who', 'they', 'it', 'golden globes 2020', 'goldenglobes2020', 'golden globe award', '#goldenglobes2020', 'globes', '@goldenglobes', 'golden globe awards', 'goldenglobe'}

def main():

    # Read in JSON data
    data = [json.loads(line) for line in open('gg2020.json','r',encoding='utf-8')]

    # Split data into two dataframes: pre-show and after show starts
    pre_data, data = split_data_by_time(data, pd.to_datetime('2020-01-06T01:00:00'))

    print(get_hosts(data))
    #print(get_winner(data))
    # print(filter_tweets(data, 'present').size)
    print(get_presenters(data))


t = time.time()
main()
print(time.time()-t)