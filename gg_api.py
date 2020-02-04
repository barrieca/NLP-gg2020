'''Version 0.35'''

import gg_func as gg

OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']

def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''

    # Get the path to the correct tweets based on the year
    data_file_path = 'gg' + str(year) + '.json'

    return gg.get_hosts_helper(data_file_path)


def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''

    # Get the path to the correct tweets based on the year
    data_file_path = 'gg' + str(year) + '.json'

    # Your code here
    return gg.get_awards_helper(data_file_path)

def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''

    # Get the path to the correct tweets based on the year
    data_file_path = 'gg' + str(year) + '.json'

    # Get the correct awards name based on the year
    award_names = OFFICIAL_AWARDS_1315 if int(year) < 2016 else OFFICIAL_AWARDS_1819

    # Your code here
    return gg.get_nominees_helper(data_file_path, award_names, year)

def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''

    # Get the path to the correct tweets based on the year
    data_file_path = 'gg' + str(year) + '.json'

    # Get the correct awards name based on the year
    award_names = OFFICIAL_AWARDS_1315 if int(year) < 2016 else OFFICIAL_AWARDS_1819

    # Your code here
    return gg.get_winner_helper(data_file_path, award_names, year)

def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''

    # Get the path to the correct tweets based on the year
    data_file_path = 'gg' + str(year) + '.json'

    # Get the correct awards name based on the year
    award_names = OFFICIAL_AWARDS_1315 if int(year) < 2016 else OFFICIAL_AWARDS_1819

    # Your code here
    return gg.get_presenters_helper(data_file_path, award_names)

def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    print("Pre-ceremony processing complete.")
    return

def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''

    get_winner(2013)
    # Sentiment Analysis - probably a better place to put this stuff
    sentiment = gg.sentiment_analysis('gg' + str(2013) + '.json')

    return

if __name__ == '__main__':
    main()
