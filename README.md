# CS 337 - Golden Globes Tweet Analyzer

This project analyzes a set of tweets pertaining to the Golden Globes and attempts to extract the
host of the show, the Golden Globes winners and nominees, the people who presented each award,
and the what the award categories actually were.

Team Members (Group 4): Cameron Barrie, Alexander Einarsson, Marko Sterbentz

The repository for this project can be found on GitHub: https://github.com/barrieca/NLP-gg2020

## Required Packages
This project utilizes a variety of Python 3 packages in order to perform its analysis of the tweets, all of which can be installed using pip. These packages are as follows:
- IMDbPy (for querying IMDb)
  - Installation: with `pip install imdbpy`
  - Webpage: https://imdbpy.github.io/
- Levenshtein (for fuzzy matching)
  - Installation: with `pip install python-Levenshtein`
  - Webpage: https://pypi.org/project/python-Levenshtein/
- Pandas (for data containers)
  - Installation: with `pip install pandas`
  - Webpage: https://pandas.pydata.org/
- Spacy (for general NLP tasks)
  - Installation: `pip install -U spacy`
  - Webpage: https://spacy.io/
- VaderSentiment (for sentiment analysis)
  - Installation: `pip install vaderSentiment`
  - Webpage: https://github.com/cjhutto/vaderSentiment

## Running the Project

TODO: instructions on what file(s) to run,


## Analysis Approach
At a high level, our approach to performing analysis for each of the goals is relatively similar.
First, the set of tweets to analyze is read into a Pandas dataframe. A set of regex filters are then
applied to each of the tweets in order to determine if a given tweet would be useful in determining
the solution to the goal at hand. The remaining tweets are then parsed and broken up into entity noun
chunks. These noun chunks are then analyzed to determine the frequency with which they occur, and a
voting system determines the entity that is most likely to be correct. There are modifications and
specialized techniques used for most of the goals, which are discussed in the following sections, but
the overarching approach is mostly the same.

### Getting Host
Since the Golden Globes have historically had at most two hosts, we use a statistical truncation method
in order to find a maximum of two hosts instead of simply reporting the top n most probably candidates.
However, since we have no prior knowledge as to the actual number of hosts, the frequency of the most
likely host is used as a basis for determining whether there was another host. We hypothesize that
the other host would likely be mentioned highly frequently in the same or similar tweets as the
most probable host.

### Getting Winner and Nominees
For this goal, we utilize IMDb in order to determine if the most likely candidates for the nominees and
winners are in fact either people or movies of which IMDb has knowledge. Once the most likely entities are
determined, a query is made to IMDb to see if this entity, or a similar entity, exists within the database.
This query utilizes fuzzy matching, which allows for some flexibility in the spelling of the candidate's
name, allowing for similar entities to be returned. Furthermore, if we are analzying the candidates associated
with an award given to movies, we require that the result be a movie and that it was released in within the
two years prior to the year of the Golden Globes show being analyzed. For a television award category, we
verify that it was released on television and that the show started within the last fifteen years prior to
the Golden Globes show being analyzed.

### Getting the Presenters
This goal utilizes techniques from the previous two goals. We assume that the presenters are all people that
would likely be within IMDb, i.e. entertainers and other artists. Accordingly, we query IMDb to ensure that
the most probable people found do indeed fit this criteria. Furthermore, since we do not know how many
people will present each award category, the statistical truncation method used for finding hosts is employed
here as well.

### Getting the Award Names
One primary problem with extracting award names from the set of tweets is that they the names themselves
tend to be very long and formal. This is at odds with the way most people tweet, and they often prefer
shorter phrases that can easily be confused for a different award category. In order to extract the award names
from the set of most probable candidates, the set of probable entities undergoes a fuzzy grouping procedure. This
groups together candidates that are similar to each other, or substrings of each other, in order to prevent
duplicate award categories from being reported as part of our final results.

### Detecting Sentiment (EC)
The sentiment_analysis_helper function is called from the main function of
autograder.py. The return value of this function is then printed. As it is
implemented, the results of this feature are meant to convey the overall
reception of tweeters to the award winners. The results are shown in the format
("winner name": "sentiment polarity" -> "actual text interpretation of polarity").

It is expected that the main of autograder.py will be called after the minimum
requirement functions are called. Specifically, this feature requires a csv file
that gets output by the get_winner function. If this csv file is not found, it
will be recreated (which will be time consuming).


### Detecting the Best, Worst, and Most Controversially Dressed (EC)
