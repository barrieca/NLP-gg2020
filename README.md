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
  - We use the `en_core_web_sm` model, which can be installed with `python -m spacy download en_core_web_sm` after installing Spacy.
  - Webpage: https://spacy.io/
- VaderSentiment (for sentiment analysis)
  - Installation: `pip install vaderSentiment`
  - Webpage: https://github.com/cjhutto/vaderSentiment
  
Note that a full list of all packages installed can be found in requirements.txt.

## Running the Project

This project can be executed by running the autograder as follows:

`python autograder.py <year>`

The runtime can be found by adding pre-pending this with the `time` command:

`time python autograder.py <year>`

This will perform the tweets analysis to find hosts, awards, nominees, presenters, winners, sentiment 
analysis of tweets about the award winners, best dressed, worst dressed, and most controversially 
dressed. The final output is provided in a human readable format via the output console/terminal.
Note that for the additional goals, two function calls were added to `main` in the autograder.
These should automatically run for the year given as input and provide output via the console/terminal

Note that this project requires an active internet connection in order to properly query the IMDb database.

### Runtime
On our computers, running the five required analyses takes roughly 7m20s. Adding the analysis for the
additional goals brings the total runtime up to roughly 8m30s, which is still below the time limit of 10m.

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

### Detecting Sentiment (Additional Goal)
The sentiment_analysis_helper function is called from the main function of
autograder.py. The return value of this function is then printed. As it is
implemented, the results of this feature are meant to convey the overall
reception of tweeters to the award winners. The results are shown in the format
("winner name": "sentiment polarity" -> "actual text interpretation of polarity").

It is expected that the main of autograder.py will be called after the minimum
requirement functions are called. Specifically, this feature requires a csv file
that gets output by the get_winner function. If this csv file is not found, it
will be recreated (which will be time consuming).

### Detecting the Best, Worst, and Most Controversially Dressed (Additional Goal)
To determine the best, worst, and most controversially dressed people in the award
show, we use sentiment analysis on clothes related tweets. To find the best and
worst dressed, we take the total sentiment score for each of the top 20 most
mentioned people in those tweets. We then weigh those scores against their total
number of mentions instead of taking a strict average, because we considered
many mentions to be indicative of good (or bad) dress. That gives each person
their own sentiment score, so we take the first and last person in a sorted
list of scores and proclaim them the best and worst dressed respectively.

We follow a similar path to find the most controversially dressed. Instead of
taking the total sentiment score, we only use the tweets that have a non-neutral
sentiment score, and label each of those tweets by their sentiment sign.
We then divide this score by their total mentions. We then take the absolute
value of those scores to find the person who has a controversy score nearest
0, as this is the person with most split sentiments.

## Randomness in the Results
For most of the goals we analyze a random sampling of the input tweets in order to ensure that the program
finishes in a timely manner. This introduces some amount of non-determinism in the final results, but the
scores we get are relatively stable and run within the allotted time on our laptops.