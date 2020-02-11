'''Version 0.35'''

import gg_func as gg
import sys

OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']

winners = {}

def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''

    # Get the path to the correct tweets based on the year
    data_file_path = 'gg' + str(year) + '.json'
    hosts = gg.get_hosts_helper(data_file_path)
    print("\nHosts\n------")
    for host in hosts:
        print('Host: ' + host)
    print('')
    return hosts


def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''

    # Get the path to the correct tweets based on the year
    data_file_path = 'gg' + str(year) + '.json'

    # Your code here
    awards = gg.get_awards_helper(data_file_path)
    print("\nAwards\n------")
    for award in awards:
        print(award)
    print('')
    return awards

def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''

    # Get the path to the correct tweets based on the year
    data_file_path = 'gg' + str(year) + '.json'

    # Get the correct awards name based on the year
    award_names = OFFICIAL_AWARDS_1315 if int(year) < 2016 else OFFICIAL_AWARDS_1819

    # Your code here
    nominees = gg.get_nominees_helper(data_file_path, award_names, year)
    print("\nNominees\n--------")
    for award in nominees.keys():
        # print(award + ': ' + ', '.join(nominees[award]))
        print(award + ': ' + str(nominees[award]))
    print('')
    return nominees

def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    global winners

    # Get the path to the correct tweets based on the year
    data_file_path = 'gg' + str(year) + '.json'

    # Get the correct awards name based on the year
    award_names = OFFICIAL_AWARDS_1315 if int(year) < 2016 else OFFICIAL_AWARDS_1819

    # Your code here
    winners[year] = gg.get_winner_helper(data_file_path, award_names, year)
    print("\nWinners\n-------")
    for award in winners[year].keys():
        print(award + ': ' + winners[year][award])
    print('')
    return winners[year]

def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''

    # Get the path to the correct tweets based on the year
    data_file_path = 'gg' + str(year) + '.json'

    # Get the correct awards name based on the year
    award_names = OFFICIAL_AWARDS_1315 if int(year) < 2016 else OFFICIAL_AWARDS_1819

    # Your code here
    presenters = gg.get_presenters_helper(data_file_path, award_names, year)
    print("\nPresenters\n----------")
    for award in presenters.keys():
        print(award + ': ' + str(presenters[award]))
    print('')
    return presenters

def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    print("Pre-ceremony processing complete.")
    return

def get_sentiment(year):

    # Sentiment Analysis
    data_file_path = 'gg' + str(year) + '.json'
    award_names = OFFICIAL_AWARDS_1315 if int(year) < 2016 else OFFICIAL_AWARDS_1819
    sentiment_dict = gg.sentiment_analysis_helper(data_file_path, award_names, year, list(winners[year].values()))
    print("\nSentiment (of winners)\n----------")
    for subject in sentiment_dict:
        print(subject + ': ' + str(sentiment_dict[subject]) + ' -> ' + gg.polarity_to_text(sentiment_dict[subject]))
    print('')

    return

def get_best_dressed(year):
    # Get the path to the correct tweets based on the year
    data_file_path = 'gg' + str(year) + '.json'

    # Your code here
    gg.get_best_dressed_helper(data_file_path, year)

    return


def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''

    return

if __name__ == '__main__':

    main()

