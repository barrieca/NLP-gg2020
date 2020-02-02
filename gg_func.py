import json
import spacy
import time
import imdb
import itertools
import Levenshtein
import pandas as pd
import re

def find_truncated_candidates(df, search_type, min=1):
    '''
    Finds a statistically truncated list of candidates that are in IMDb.
    :param df: DataFrame of sorted nouns
    :param search_type: 'name' for person(s) or 'title' for movie(s)
    :param min: The number of candidates (default is 1).
    :return: The list of candidates.
    '''
    return statistical_truncation(find_imdb_objects(df, search_type, min), 0.5, min)

def is_valid_movie_year(test_year, award_year):
    return test_year != '????' and int(test_year) >= int(award_year)-2 and int(test_year) < int(award_year)

def is_valid_series_year(test_year, award_year):
    return test_year != '????' and int(test_year) >= int(award_year)-15 and int(test_year) < int(award_year)

def find_imdb_objects(df, search_type, n=1, year=0, is_movie=False, fuzzy_threshold=0.25):
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
        if not any(possible[search_type] in results_elt for results_elt in results for possible in result): # get rid of duplicates
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

def filter_tweets(df_tweets, regex_string, invert=False):
    '''
    Filters the dataframe of tweet text to only include those that match the given regex_string.
    :param df_tweets: The dataframe containing the tweets' text
    :param regex_string: The regular expression string to apply to each tweet (i.e. movie|film|picture)
    :return:
    Example: noun_df = filter_tweets(pd.DataFrame(tweet_list, columns=['text']), 'movie|film|picture')
    '''

    return df_tweets[~df_tweets.text.str.contains(regex_string, regex=True)] if invert\
        else df_tweets[df_tweets.text.str.contains(regex_string, regex=True)]

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
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'direct')
    if 'cecil' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'cecil')
    if 'carol' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'carol')
    if 'animate' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'animate')
    if 'foreign' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'foreign|language')
    if 'screenplay' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'screen|write')
    if 'score' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'score')
    if 'song' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'song')

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

def trimend(string, pattern):
    idx = re.search(pattern, string)
    if idx:
        string = string[0:idx.start()]
    return string

def fuzzy_group(df_phrases, truncate_at):
    phrase_list = []
    for i, row in df_phrases.iterrows():
        if len(phrase_list) >= truncate_at:
            break
        # (row['text'] = trimend(row['text'], x)) for x in [' for', ' goes to']
        row['text'] = trimend(row['text'], ' for( |$)')
        row['text'] = trimend(row['text'], ' goes to')
        row['text'] = trimend(row['text'], ' in$')
        row['text'] = trimend(row['text'], ' at the')
        already_in_list = False
        for j in range(len(phrase_list)):
            if phrase_list[j].startswith(row['text']):
                already_in_list = True
                break
            elif row['text'].startswith(phrase_list[j]):
                phrase_list[j] = row['text']
                already_in_list = True
                break
        if not already_in_list and row['text']:
            phrase_list.append(row['text'])
    return phrase_list

def search_for_awards(df_tweets):

    # g1 = df_tweets['text'].str.extract(r'(?:wins|won) (best [\w ,-]+) (?:for|[!#])', re.IGNORECASE).dropna()
    # g2 = df_tweets['text'].str.extract(r'(?:wins|won)[\w ]*golden ?globe[\w ]* (?:for|[!#]) (best [\w ,-]+)', re.IGNORECASE).dropna() # this may not work well
    # g3 = df_tweets['text'].str.extract(r'winner of (best [^!#]+) is', re.IGNORECASE).dropna() # this may not work well
    # g4 = df_tweets['text'].str.extract(r'present(?:s|ed) (best [\w ,-]+)', re.IGNORECASE).dropna()
    # g4 = df_tweets['text'].str.extract(r'presents (best [\w ,-]+)', re.IGNORECASE).dropna()
    # g4 = df_tweets['text'].str.extract(r'presented (best [\w ,-]+)', re.IGNORECASE).dropna()
    # g5 = df_tweets['text'].str.extract(r' ?(best [\w ,-]+) goes to', re.IGNORECASE).dropna()

    # phrases = pd.concat([g1, g2, g3, g4, g5])
    phrases = df_tweets['text'].str.extract(r'for (best [\w ,-]+) (?:for|[!#])', re.IGNORECASE).dropna()
    phrases = phrases[~phrases[0].str.contains('golden ?globe', case=False, regex=True)]
    return pd.DataFrame([candidate.lower() for candidate in phrases[0]], columns=['text'])

def get_noun_frequencies(df_nouns):
    '''
    Returns sorted data frame of unique, candidate nouns.
    :param df_nouns:
    :return:
    '''
    df_nouns['freq'] = df_nouns.groupby('text')['text'].transform('count')
    return df_nouns.drop_duplicates().sort_values(by='freq', ascending=False)

def statistical_truncation(list_candidates, threshold_percent=0.6, min = 0):
    '''
    :param list_candidates:
    :param threshold_percent:
    :param min:
    :return: List of answers.
    Example Usage:
    tup_list = [('John',500),('Jane',450),('Jim',400),('Jake',399),('Jesse',300)]
    statistical_truncation(tup_list,0.8) = ['John', 'Jane', 'Jim']
    '''
    # print(list_candidates)
    try:
        top_frequency = list_candidates[0][1]
        result_list = []
        for candidate in list_candidates:
            if candidate[1] < top_frequency * threshold_percent and len(result_list) >= min:
                break
            else:
                result_list.append(candidate[0])
        return result_list
    except:
        return []

def split_data_by_time(json_data, start_time):
    '''
    Builds two dataframes of the tweets with tweets before and after the starting time of the GG.
    :param json_data: The JSON data
    :param start_time: The start time of the Golden Globes in UTC datetime format.
    :return: Two dataframes of tweets (pre-show and non-pre-show) with desirable qualities.
    '''

    df_tweets = pd.DataFrame(json_data[0], columns=['text'])        # Indexing [0] will cause problems for 2020 data

    if 'created_at' in df_tweets:
        df_tweets_after_start = df_tweets[pd.to_datetime(df_tweets['created_at']) >= start_time]
        df_tweets_before_start = df_tweets[pd.to_datetime(df_tweets['created_at']) < start_time]
        return df_tweets_before_start[['text']], df_tweets_after_start[['text']]
    else:
        df_tweets_before_start = pd.DataFrame()
        df_tweets_after_start = df_tweets[['text']]
        return df_tweets_before_start, df_tweets_after_start

def get_hosts_helper(data_file_path):
    '''
    Filters tweets containing 'host' to find, aggregates, and sorts names in the imdb database.
    :param data_file_path: Path to the JSON file of tweets.
    :return:
    '''
    # print("finding host")
    # Read in JSON data
    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    # Split data into two dataframes: pre-show and after show starts
    pre_data, data = split_data_by_time(json_data, pd.to_datetime('2020-01-06T01:00:00'))

    # Filter the tweets to find ones containing references to host
    df_filtered_tweets = filter_tweets(data, 'host')
    df_filtered_tweets = filter_tweets(df_filtered_tweets, 'next', True)
    max_hosts = 2
    # df_filtered_tweets = filter_tweets(df_filtered_tweets, 'next', True)
    # print("filtered host tweets | " + str(df_filtered_tweets.size) + " remaining")

    # Get the entities present in these tweets
    df_filtered_tweets = get_noun_frequencies(create_noun_chunks(df_filtered_tweets))
    # print('found possible host entities')

    # Determine the most likely host, max 2 hosts
    return find_truncated_candidates(df_filtered_tweets, 'name', max_hosts)

def get_awards_helper(data_file_path):
    '''

    :param data_file_path: Path to the JSON file of tweets.
    :return:
    '''
    # Read in JSON data
    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    df_tweets = pd.DataFrame(json_data[0], columns=['text'])

    df_nominee_tweets = filter_tweets(df_tweets, 'win|won|goes to|congratulations|congrats|congratz')

    df_candidates = search_for_awards(df_nominee_tweets)

    # df_noun_chunks = create_noun_chunks(df_nominee_tweets)
    df_sorted_nouns = get_noun_frequencies(df_candidates)
    phrases = fuzzy_group(df_sorted_nouns, 27)
    # temp = df_sorted_nouns['text'][0:27]
    return phrases

def get_nominees_helper(data_file_path, award_names, awards_year):
    '''
    Determines the winner for each award based on dataset of tweets.
    :param data_file_path: Path to the JSON file of tweets.
    :param award_names: The award names for the current year.
    :param awards_year: The year the Golden Globes were held.
    :return: A dictionary with the hard coded award names as keys, and each entry a list of strings denoting nominees.
    '''

    # Define some useful parameters for processing
    num_possible_winner = 5
    award_nominees = {}
    award_entity_type = dict(map(entity_typer, award_names))

    # Read in JSON data
    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    # Split data into two dataframes: pre-show and after show starts
    pre_data, data = split_data_by_time(json_data, pd.to_datetime('2020-01-06T01:00:00'))

    t = time.time()

    # For each award category
    for category in award_names:
        # Filter tweets by subject string
        df_nominee_tweets = filter_tweets(data, 'nomin|should|wish|win|won|goes to|not win|nod|sad|pain|down|hope|rob|snub')

        # Filter based on the award category
        df_nominee_tweets = filter_by_category(df_nominee_tweets, category)

        print("filtered nominee tweets | " + str(df_nominee_tweets.size))

        # Get the nouns chunks in the remaining tweets
        df_noun_chunks = create_noun_chunks(df_nominee_tweets)
        # print("found noun chunks")

        # Aggregate and sort the noun chunks
        df_sorted_nouns = get_noun_frequencies(df_noun_chunks)
        # print("found noun frequencies")

        # Produce the correct number of noun chunks that also exist on IMDb
        imdb_candidates = find_imdb_objects(df_sorted_nouns, entity_type_to_imdb_type[award_entity_type[category]], num_possible_winner, awards_year, award_entity_type[category] == 'movie')
        # print("found imdb candidates")

        # Store winner
        award_nominees[category] = [nominee[0] for nominee in imdb_candidates[0:num_possible_winner]]
        # print("found the award winner")

    print(award_nominees)
    print(t - time.time())
    return award_nominees

def get_presenters_helper(data_file_path, award_names):
    '''
    Determines the winner for each award based on dataset of tweets.
    :param data_file_path: Path to the JSON file of tweets.
    :param award_names: The award names for the current year.
    :param awards_year: The year the Golden Globes were held.
    :return: A dictionary with the hard coded award names as keys, and each entry a list of strings denoting nominees.
    '''

    # Define some useful parameters for processing
    num_possible_presenters = 2
    award_presenters = {}
    award_entity_type = dict(map(entity_typer, award_names))

    # Read in JSON data
    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    # Split data into two dataframes: pre-show and after show starts
    pre_data, data = split_data_by_time(json_data, pd.to_datetime('2020-01-06T01:00:00'))

    t = time.time()

    # For each award category
    for category in award_names:
        # Filter tweets by subject string
        df_nominee_tweets = filter_tweets(data, 'present|giv|hand|introduc')

        # Filter based on the award category
        df_nominee_tweets = filter_by_category(df_nominee_tweets, category)

        print("filtered presenter tweets | " + str(df_nominee_tweets.size))

        # Get the nouns chunks in the remaining tweets
        df_noun_chunks = create_noun_chunks(df_nominee_tweets)
        # print("found noun chunks")

        # Aggregate and sort the noun chunks
        df_sorted_nouns = get_noun_frequencies(df_noun_chunks)
        # print("found noun frequencies")

        # Produce the correct number of noun chunks that also exist on IMDb
        imdb_candidates = find_imdb_objects(df_sorted_nouns, entity_type_to_imdb_type[award_entity_type[category]], num_possible_presenters)
        # print("found imdb candidates")

        # Store winner
        award_presenters[category] = statistical_truncation(imdb_candidates, 0.6, 1)
        # print("found the award presenters")

    print(award_presenters)
    print(t - time.time())
    return award_presenters

def get_winner_helper(data_file_path, award_names, awards_year):
    '''
    Determines the winner for each award based on dataset of tweets.
    :param data_file_path: Path to the JSON file of tweets.
    :param award_names: The award names for the current year.
    :param awards_year: The year the Golden Globes were held.
    :return: Dictionary containing 27 keys, with list as its value
    '''

    # Define some useful parameters for processing
    num_possible_winner = 1
    award_winners = {}
    award_entity_type = dict(map(entity_typer, award_names))

    # Read in JSON data
    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    # Split data into two dataframes: pre-show and after show starts
    pre_data, data = split_data_by_time(json_data, pd.to_datetime('2020-01-06T01:00:00'))

    # For each award category
    for category in award_names:
        # t = time.time()
        # Filter tweets by subject string
        df_nominee_tweets = filter_tweets(data, 'win|won|goes to|congratulations|congrats|congratz')

        # Filter based on the award category
        df_nominee_tweets = filter_by_category(df_nominee_tweets, category)

        print("filtered winner tweets | " + str(df_nominee_tweets.size))

        # Get the nouns chunks in the remaining tweets
        df_noun_chunks = create_noun_chunks(df_nominee_tweets)
        # print("found noun chunks")

        # Aggregate and sort the noun chunks
        df_sorted_nouns = get_noun_frequencies(df_noun_chunks)
        # print("found noun frequencies")

        # Produce the correct number of noun chunks that also exist on IMDb
        imdb_candidates = find_imdb_objects(df_sorted_nouns, entity_type_to_imdb_type[award_entity_type[category]], num_possible_winner, awards_year, award_entity_type[category] == 'movie')
        # print("found imdb candidates")

        # Store winner
        award_winners[category] = imdb_candidates[0][0]
        # print("found the award winner")
        # print(t-time.time())

    return award_winners

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
        return (award_name, 'movie') # this should probably be song
    elif 'series' in award_name:
        return (award_name, 'tv')
    else:
        return (award_name, 'movie')

entity_type_to_imdb_type = {'person': 'name', 'tv': 'title', 'movie': 'title'}

noun_chunk_stop_words = {'i', 'you', 'golden globe', 'golden globes', 'goldenglobes', 'congratulations', '#', 'the golden globes', 'a golden globe', 'the golden globe', 'he', 'she', 'me', 'who', 'they', 'it', 'golden globes 2020', 'goldenglobes2020', 'golden globe award', '#goldenglobes2020', 'globes', '@goldenglobes', 'golden globe awards', 'goldenglobe'}

# def main():

    # Read in JSON data
    # data = [json.loads(line) for line in open('gg2020.json','r',encoding='utf-8')]
    # print(type(data[0]))
    # # Split data into two dataframes: pre-show and after show starts
    # pre_data, data = split_data_by_time(data, pd.to_datetime('2020-01-06T01:00:00'))
    #
    # print(get_hosts_helper(data))

#     #print(get_winner(data))
#     # print(filter_tweets(data, 'present').size)
#     print(get_presenters_helper(data))


# t = time.time()
# main()
# print(time.time()-t)
