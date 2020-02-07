Readme information for extra credit

Sentiment Analysis:
################################################################################
The sentiment_analysis_helper function is called from the main function of
autograder.py. The return value of this function is then printed. As it is
implemented, the results of this feature are meant to convey the overall
reception of tweeters to the award winners. The results are shown in the format
(<winner name>: <sentiment polarity> -> <actual text interpretation of polarity>).

It is expected that the main of autograder.py will be called after the minimum
requirement functions are called. Specifically, this feature requires a csv file
that gets output by the get_winner function. If this csv file is not found, it
will be recreated (which will be time consuming).

