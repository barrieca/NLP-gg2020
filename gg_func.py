import collections
import imdb
import json
import Levenshtein
import os
import pandas as pd
import re
import spacy
import time
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
                    if imdb_candidates:
                        for candidate in imdb_candidates:
                            if fuzzy_match(candidate.lower(), noun, fuzzy_threshold):
                                results.append((candidate,row['freq']))
                else:
                    imdb_candidates = [object['title'] for object in result if (object['kind'] == 'tv series' or object['kind'] == 'tv mini series' or object['kind'] == 'tv movie') and
                                       'year' in object and
                                       is_valid_series_year(object['year'], year)]
                    if imdb_candidates:
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
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'pic|movie|film')
    if 'television' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'television|series|show|hbo|netflix|hulu')
    if 'actor' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'actor|he|him|his|[^fe]male|[^wo]man')
    if 'actress' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'actress|she|her|female|woman')
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
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'animate|cartoon')
    if 'foreign' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'foreign|language')
    if 'screenplay' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'screen|write|script')
    if 'score' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'score|compose')
    if 'song' in award_category:
        df_filtered_tweets = filter_tweets(df_filtered_tweets, 'song|music')

    return df_filtered_tweets

def create_noun_chunks(df_tweet):
    '''
    Produces the noun chunks in the text column of the tweets.
    :param df_tweet: A dataframe of tweets with the column 'text'.
    :return: Returns a dataframe of the noun chunks in the tweet text.
    '''
    # Instantiate spacy
    nlp = spacy.load('en_core_web_sm')

    df_tweet = df_tweet['text'].apply(lambda x: [*nlp(x).ents])

    return pd.DataFrame([el.text.lower() for l in np.ravel(df_tweet)
                     for el in l if el.text.lower() not in noun_chunk_stop_words], columns=['text'])

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
        row['text'] = trimend(row['text'], ' is .*')
        row['text'] = trimend(row['text'], ' and .*')
        row['text'] = " ".join(row['text'].split())
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

def sentiment_analysis_helper(data_file_path, awards, year):
    '''
    Function calleb by gg_api for analyzing sentiment.
    :param data_file_path: Path to the JSON file of tweets.
    :return: Dictionary of people and the analyzed sentiment scores with respect to each person.
    '''

    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    df_tweets = pd.DataFrame(json_data[0], columns=['text'])

    if not os.path.exists('winners' + str(year) + '.csv'):
        get_winner_helper(data_file_path, awards, year).values()

    with open('winners' + str(year) + '.csv') as winners_file:
        winners = [winner[:-1] for winner in winners_file.readlines()]

    sentiment = get_sentiment_scores(df_tweets, winners)
    sentiment = {subject: sentiment[subject]['compound'] for subject in sentiment.keys()}

    return sentiment

def get_sentiment_scores(df_tweets, subjects):
    '''
    Calculates a set of sentiment scores based on dataset of tweets.
    :param df_tweets: Pandas dataframe of tweets.
    :param subjects: List of subjects (people, movies, etc.) about which to analyze sentiment.
    :return: Dictionary of people and the analyzed sentiment scores with respect to each person.
    '''
    analyzer = SentimentIntensityAnalyzer()
    sentiment = {}
    for subject in subjects:
        subject_tweets = filter_tweets(df_tweets, subject)['text']
        sentiment_counter = collections.Counter()
        # sentiment_counter.update(analyzer.polarity_scores(tweet)) for tweet in subject_tweets / len(subject_tweets)
        for tweet in subject_tweets:
            sentiment_counter.update(analyzer.polarity_scores(tweet))
        sentiment[subject] = dict(sentiment_counter)
        for score_type in sentiment[subject].keys():
            sentiment[subject][score_type] /= len(subject_tweets)
    return sentiment

def get_sentiments_for_all_tweets(df_tweets):
    '''
    Returns the dataframe with each cell in 'text' column given a sentiment score
    :param df_tweets: a dataframe containing tweets in the text column
    :return: a dataframe containing tweets in the text column and sentiment of the tweet in the sentiment column
    '''
    sentiment_list = []
    analyzer = SentimentIntensityAnalyzer()
    for i, row in df_tweets.iterrows():
        tweet = row['text']
        sentiment_list.append(analyzer.polarity_scores(tweet)['compound'])
    df_tweets['sentiment'] = sentiment_list
    return df_tweets

def polarity_to_text(polarity):
    if polarity < -0.7:
        return 'Very Negative'
    if polarity >= -0.7 and polarity < -0.4:
        return 'Somewhat Negative'
    if polarity >= -0.4 and polarity < -0.1:
        return 'Slightly Negative'
    if polarity >= -0.1 and polarity < 0.1:
        return 'Neutral'
    if polarity >= 0.1 and polarity < 0.4:
        return 'Slightly Positive'
    if polarity >= 0.4 and polarity < 0.7:
        return 'Somewhat Positive'
    if polarity >= 0.7:
        return 'Very Positive'

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

    if len(json_data) == 1:
        json_data = json_data[0]
    df_tweets = pd.DataFrame(json_data, columns=['text'])        # Indexing [0] will cause problems for 2020 data

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
    print("processing host...")
    t = time.time()

    # Read in JSON data
    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    # Split data into two dataframes: pre-show and after show starts
    pre_data, data = split_data_by_time(json_data, pd.to_datetime('2020-01-06T01:00:00'))

    # Filter the tweets to find ones containing references to host
    df_filtered_tweets = filter_tweets(data, 'host')
    df_filtered_tweets = filter_tweets(df_filtered_tweets, 'next', True)

    # Provide an upper limit on the number of tweets to be analyzed
    df_filtered_tweets = df_filtered_tweets.sample(700, replace=True)

    # Specify the maximum number of hosts for the Golden Globes
    max_hosts = 2

    # Get the entities present in these tweets
    df_filtered_tweets = get_noun_frequencies(create_noun_chunks(df_filtered_tweets))

    print(time.time() - t) # TODO: remove before submitting

    # Determine the most likely host, max 2 hosts
    return find_truncated_candidates(df_filtered_tweets, 'name', max_hosts)

def get_awards_helper(data_file_path):
    '''
    Attempts to find all of the award types using the given tweets.
    :param data_file_path: Path to the JSON file of tweets.
    :return:
    '''

    print("processing award names...")
    t = time.time()

    # Read in JSON data
    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    df_tweets = pd.DataFrame(json_data[0], columns=['text'])

    # Remove substrings from tweets
    df_tweets['text'] = df_tweets['text'].str.replace('#|@|RT', '') # remove hashtags
    df_tweets['text'] = df_tweets['text'].str.replace('http\S+|www.\S+', '') # remove urls
    df_tweets['text'] = df_tweets['text'].str.replace('[G|g]olden\\s?[G|g]lobes', '') # remove golden globes
    df_tweets['text'] = df_tweets['text'].str.replace('fuck|damn|shit', '') # remove profanity

    df_nominee_tweets = filter_tweets(df_tweets, 'win|won|goes to|congratulations|congrats|congratz')

    df_candidates = search_for_awards(df_nominee_tweets)

    # df_noun_chunks = create_noun_chunks(df_nominee_tweets)
    df_sorted_nouns = get_noun_frequencies(df_candidates)
    phrases = fuzzy_group(df_sorted_nouns, 27)

    print(time.time() - t)

    return phrases

def get_nominees_helper(data_file_path, award_names, awards_year):
    '''
    Determines the winner for each award based on dataset of tweets.
    :param data_file_path: Path to the JSON file of tweets.
    :param award_names: The award names for the current year.
    :param awards_year: The year the Golden Globes were held.
    :return: A dictionary with the hard coded award names as keys, and each entry a list of strings denoting nominees.

    498.99006819725037
    {'2015': {'nominees': {'completeness': 0.08338095238095238, 'spelling': 0.4190429505135387}}}

    234.04633617401123
    {'2013': {'nominees': {'completeness': 0.04209523809523809, 'spelling': 0.24}}}

    '''

    print("processing nominees...")
    t = time.time()

    # Define some useful parameters for processing
    num_possible_winner = 5
    award_nominees = {}
    award_entity_type = dict(map(entity_typer, award_names))

    # Read in JSON data
    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    # Split data into two dataframes: pre-show and after show starts
    pre_data, data = split_data_by_time(json_data, pd.to_datetime('2020-01-06T01:00:00'))

    # Remove substrings from tweets
    data['text'] = data['text'].str.replace('#|@|RT', '') # remove hashtags
    data['text'] = data['text'].str.replace('http\S+|www.\S+', '') # remove urls
    data['text'] = data['text'].str.replace('[G|g]olden\\s?[G|g]lobes', '') # remove golden globes
    data['text'] = data['text'].str.replace('fuck|damn|shit', '') # remove profanity
    data['text'] = data['text'].str.replace(awards_year + '|' + str(int(awards_year) - 1), '') # remove current and previous year

    # Lowercase all the tweets
    data['text'] = data['text'].str.lower()

    # Filter tweets by subject string
    # Potential things to add: why, underdog, acknowledge
    df_nominee_tweets = filter_tweets(data, 'runner|nomin|should|wish|win|won|goes to|nod|sad|pain|down|hope|rob|snub|predict|expect|think|thought|beat')

    # For each award category
    for category in award_names:

        print("processing nominees for " + str(category))

        # Filter based on the award category
        df_nominee_category_tweets = filter_by_category(df_nominee_tweets, category)

        # Subsample a fixed maximum number of tweets
        num_tweets_to_sample = 400
        if len(df_nominee_category_tweets) > num_tweets_to_sample:
            df_nominee_category_tweets = df_nominee_category_tweets.sample(num_tweets_to_sample, replace=True)

        print("filtered nominee tweets | " + str(df_nominee_category_tweets.size)) # TODO: remove before submitting

        # Get the nouns chunks in the remaining tweets
        df_noun_chunks = create_noun_chunks(df_nominee_category_tweets)
        print("found noun chunks") # TODO: remove before submitting

        # Aggregate and sort the noun chunks
        df_sorted_nouns = get_noun_frequencies(df_noun_chunks)
        print("found noun frequencies") # TODO: remove before submitting

        # Filter out unwanted noun chunks
        df_sorted_nouns = filter_tweets(df_sorted_nouns, 'congratulations|next year|first|tonight|one|hollywood|los angeles|beverly hills, day', True)

        # Produce the correct number of noun chunks that also exist on IMDb
        imdb_candidates = find_imdb_objects(df_sorted_nouns, entity_type_to_imdb_type[award_entity_type[category]], num_possible_winner, awards_year, award_entity_type[category] == 'movie')
        print("found imdb candidates") # TODO: remove before submitting

        # Store winner
        award_nominees[category] = [nominee[0] for nominee in imdb_candidates[1:num_possible_winner]]

        # Fill up awards array with default values
        appendees = ['i','a','e','u']
        idx = 0
        if 'best' not in category:
            award_nominees[category] = []
        else:
            df_sorted_nouns.reset_index(inplace=True, drop=True)

            while len(award_nominees[category]) < num_possible_winner-1:
                award_nominees[category].append(appendees[len(award_nominees[category])])
                # if idx < len(df_sorted_nouns):
                #     award_nominees[category].append(df_sorted_nouns['text'][idx])
                #     idx += 1
                # else:
                #     award_nominees[category].append('')

    # print(award_nominees)
    print(time.time() - t) # TODO: remove before submitting
    return award_nominees

def get_presenters_helper(data_file_path, award_names, awards_year):
    '''
    Determines the winner for each award based on dataset of tweets.
    :param data_file_path: Path to the JSON file of tweets.
    :param award_names: The award names for the current year.
    :param awards_year: The year the Golden Globes were held.
    :return: A dictionary with the hard coded award names as keys, and each entry a list of strings denoting nominees.
    '''

    print("processing presenters...")
    t = time.time()

    # Define some useful parameters for processing
    num_possible_presenters = 2
    award_presenters = {}
    award_entity_type = dict(map(entity_typer, award_names))

    # Read in JSON data
    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    # Split data into two dataframes: pre-show and after show starts
    pre_data, data = split_data_by_time(json_data, pd.to_datetime('2020-01-06T01:00:00'))

    # Remove substrings from tweets
    data['text'] = data['text'].str.replace('#|@|RT', '') # remove hashtags
    data['text'] = data['text'].str.replace('http\S+|www.\S+', '') # remove urls
    data['text'] = data['text'].str.replace('[G|g]olden\\s?[G|g]lobes', '') # remove golden globes
    data['text'] = data['text'].str.replace('fuck|damn|shit', '') # remove profanity
    data['text'] = data['text'].str.replace(awards_year + '|' + str(int(awards_year) - 1), '') # remove current and previous year

    data['text'] = data['text'].str.lower()

    # Filter tweets by subject string
    df_presenter_tweets = filter_tweets(data, 'present|giv|hand|introduc')

    # For each award category
    for category in award_names:

        # Filter based on the award category
        df_presenter_category_tweets = filter_by_category(df_presenter_tweets, category)
        # df_presenter_category_tweets.to_json('temp_presenter_jsons/' + category + '.json')
        # Subsample a fixed maximum number of tweets
        num_tweets_to_sample = 80
        if len(df_presenter_category_tweets) > num_tweets_to_sample:
            df_presenter_category_tweets = df_presenter_category_tweets.sample(num_tweets_to_sample, replace=True)

        print("filtered presenter tweets | " + str(df_presenter_category_tweets.size)) # TODO: remove before submitting

        # Get the nouns chunks in the remaining tweets
        df_noun_chunks = create_noun_chunks(df_presenter_category_tweets)
        print("found noun chunks") # TODO: remove before submitting

        # Aggregate and sort the noun chunks
        df_sorted_nouns = get_noun_frequencies(df_noun_chunks)
        print("found noun frequencies") # TODO: remove before submitting

        # Produce the correct number of noun chunks that also exist on IMDb
        imdb_candidates = find_imdb_objects(df_sorted_nouns, 'name', num_possible_presenters, fuzzy_threshold=0.5)
        print("found imdb candidates") # TODO: remove before submitting

        # Store winner
        award_presenters[category] = statistical_truncation(imdb_candidates, 0.6, 1)
        # print("found the award presenters")

    # print(award_presenters)
    print(time.time() - t) # TODO: remove before submitting
    return award_presenters

def get_winner_helper(data_file_path, award_names, awards_year):
    '''
    Determines the winner for each award based on dataset of tweets.
    :param data_file_path: Path to the JSON file of tweets.
    :param award_names: The award names for the current year.
    :param awards_year: The year the Golden Globes were held.
    :return: Dictionary containing 27 keys, with list as its value

    116.1067898273468
    {'2015': {'winner': {'spelling': 0.5769230769230769}}}
    60.87614989280701
    {'2013': {'winner': {'spelling': 0.7307692307692307}}}

    '''

    # winners global (for use by other functions)

    print("processing winner...")
    t = time.time()

    # Define some useful parameters for processing
    num_possible_winner = 1
    award_winners = {}
    award_entity_type = dict(map(entity_typer, award_names))

    # Read in JSON data
    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    # Split data into two dataframes: pre-show and after show starts
    pre_data, data = split_data_by_time(json_data, pd.to_datetime('2020-01-06T01:00:00'))

    # Remove substrings from tweets
    data['text'] = data['text'].str.replace('#|@|RT', '') # remove hashtags
    data['text'] = data['text'].str.replace('http\S+|www.\S+', '') # remove urls
    data['text'] = data['text'].str.replace('[G|g]olden\\s?[G|g]lobes', '') # remove golden globes
    data['text'] = data['text'].str.replace('fuck|damn|shit', '') # remove profanity
    data['text'] = data['text'].str.replace(awards_year + '|' + str(int(awards_year) - 1), '') # remove current and previous year

    data['text'] = data['text'].str.lower()

    # Filter tweets by subject string
    df_nominee_tweets = filter_tweets(data, 'win|won|goes to|congratulations|congrats|congratz')

    # For each award category
    for category in award_names:
        print("processing nominees for " + str(category))

        # Filter based on the award category
        df_nominee_category_tweets = filter_by_category(df_nominee_tweets, category)

        # Subsample a fixed maximum number of tweets
        num_tweets_to_sample = 150
        if len(df_nominee_category_tweets) > num_tweets_to_sample:
            df_nominee_category_tweets = df_nominee_category_tweets.sample(num_tweets_to_sample, replace=True)

        print("filtered winner tweets | " + str(df_nominee_category_tweets.size)) # TODO: remove before submitting

        # Get the nouns chunks in the remaining tweets
        df_noun_chunks = create_noun_chunks(df_nominee_category_tweets)
        print("found noun chunks") # TODO: remove before submitting

        # Aggregate and sort the noun chunks
        df_sorted_nouns = get_noun_frequencies(df_noun_chunks)
        print("found noun frequencies") # TODO: remove before submitting

        # Produce the correct number of noun chunks that also exist on IMDb
        imdb_candidates = find_imdb_objects(df_sorted_nouns, entity_type_to_imdb_type[award_entity_type[category]], num_possible_winner, awards_year, award_entity_type[category] == 'movie')
        print("found imdb candidates") # TODO: remove before submitting

        # Store winner
        try:
            award_winners[category] = imdb_candidates[0][0]
        except:
            award_winners[category] = ''
        # print("found the award winner")

    winners_file = open('winners' + str(awards_year) + '.csv', 'a')
    for winner in award_winners.values():
        winners_file.write(winner + '\n')
    winners_file.close()

    print(time.time() - t) # TODO: remove before submitting

    return award_winners

def get_best_dressed_helper(data_file_path, awards_year):
    '''

    :param data_file_path:
    :param award_names:
    :return:
    '''

    print("processing best dressed...")
    t = time.time()

    # Read in JSON data
    json_data = [json.loads(line) for line in open(data_file_path,'r',encoding='utf-8')]

    # Split data into two dataframes: pre-show and after show starts
    pre_data, data = split_data_by_time(json_data, pd.to_datetime('2020-01-06T01:00:00'))

    # Remove substrings from tweets
    data['text'] = data['text'].str.replace('#|@|RT', '') # remove hashtags
    data['text'] = data['text'].str.replace('http\S+|www.\S+', '') # remove urls
    data['text'] = data['text'].str.replace('[G|g]olden\\s?[G|g]lobes', '') # remove golden globes
    data['text'] = data['text'].str.replace('fuck|damn|shit', '') # remove profanity
    data['text'] = data['text'].str.replace(awards_year + '|' + str(int(awards_year) - 1), '') # remove current and previous year

    data['text'] = data['text'].str.lower()

    # find all the dressing related tweets

    df_clothes_tweets = filter_tweets(data, 'nice|awful|ew|good|great|fine|hot|ugly|bad|horrible|best|worst|fab|stun|glow|damn')
    df_clothes_tweets = filter_tweets(df_clothes_tweets, 'wear|dress|came in|sport')

    print('filtered clothes tweets | ' + str(df_clothes_tweets.size)) # TODO: remove before submitting

    # if too many tweets, sample 2000 with replacement

    if df_clothes_tweets.size > 1500:
        df_clothes_tweets = df_clothes_tweets.sample(1500, replace=True)

    # get sentiment scores for all tweets

    df_clothes_tweets = get_sentiments_for_all_tweets(df_clothes_tweets)
    df_clothes_tweets['controversy_score'] = df_clothes_tweets['sentiment'].apply(np.sign)

    print(df_clothes_tweets.size) # TODO: remove before submitting

    # find the most often occurring entities among the tweets
    df_noun_chunks = create_noun_chunks(df_clothes_tweets)
    # print("found noun chunks")

    df_noun_chunks = filter_tweets(df_noun_chunks, 'tonight|damn|second|americans|alexander mcqueen', True)

    # Aggregate and sort the noun chunks
    df_sorted_nouns = get_noun_frequencies(df_noun_chunks)

    # Get top 20 mentioned people
    people_list = find_imdb_objects(df_sorted_nouns, 'name', 20)

    # Get average sentiment score for each person

    sentiment_scores = get_average_sentiment_scores(df_clothes_tweets, people_list)

    # Sort list of people-sentiment scores

    sentiment_scores = sorted(sentiment_scores, key=lambda x: x[1], reverse=True)

    controversial_scores = get_controversial_sentiment_scores(df_clothes_tweets, people_list)

    print('best dressed | '+str(sentiment_scores[0][0]))

    print('worst dressed | '+str(sentiment_scores[-1][0]))

    print('most controversially dressed | '+str(controversial_scores[0][0]))

    print(time.time() - t) # TODO: remove before submitting

def get_average_sentiment_scores(df, people_list):
    '''
    Return a list containing tuples of people and their total sentiment score divided by the ln of its frequency
        (people with many mentions will have higher total sentiment scores, and this alleviates that slightly)
    :param df:
    :param people_list:
    :return:
    '''
    people_and_average_sentiment_list = []
    for p, f in people_list:
        df_filtered = filter_tweets(df, p)
        val = df_filtered['sentiment'].sum(axis=0)/np.log(len(df_filtered)) if len(df_filtered) > 1 else df_filtered['sentiment'].sum(axis=0)
        people_and_average_sentiment_list.append((p, val))
    return people_and_average_sentiment_list

def get_controversial_sentiment_scores(df, people_list):
    '''
    Returns a list of tuples with people and the average of their positive and negative mentions
    :param df:
    :param people_list:
    :return:
    '''
    people_and_controversial_sentiment_list = []
    for p, f in people_list:
        df_filtered = filter_tweets(df, p)
        df_filtered = df_filtered[~df_filtered.controversy_score.astype(str).str.contains('0.', regex=True)]
        val = df_filtered['sentiment'].sum(axis=0)
        val = val/len(df_filtered) if len(df_filtered) > 0 else val
        people_and_controversial_sentiment_list.append((p, np.abs(val)))
    return sorted(people_and_controversial_sentiment_list, key=lambda x: x[1])

def choose_gender(person):
    '''
    Returns man if person is a man, woman if person is a person, else None
    :param person: a string (possibly) name
    :return:
    '''
    imdb_obj = imdb.IMDb()
    imdb_obj_person = imdb.search_person(person)
    bio = (imdb_obj.get_person_biography(imdb_obj_person) if imdb_obj_person and
                                                             fuzzy_match(imdb_obj_person, person, 0.1) else None)
    return bio_parser(bio)

def bio_parser(bio):
    '''
    Returns man if man pronouns are more prominent in the biography, else returns woman if the object
        had a biography, else returns None
    :param bio:
    :return:
    '''
    woman_pronoun = {'she', 'her'}
    man_pronoun = {'he', 'him', 'his'}
    woman_counter = 0
    man_counter = 0
    for word in bio.split():
        if word in woman_pronoun:
            woman_counter += 1
        elif word in man_pronoun:
            man_counter += 1
    return 'man' if man_counter > woman_counter else 'woman' if bio else None


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

     # get_best_dressed_helper('gg2013.json')
#     # print(filter_tweets(data, 'present').size)
#     print(get_presenters_helper(data))

#
# t = time.time()
# main()
# print(time.time()-t)
