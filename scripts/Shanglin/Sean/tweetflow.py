# File: tweetflow.py
# Author: Sam Shanglin Yang
# Contact: syang662@wisc.edu

import pandas as pd

class Tweetflow:
    """
    A helper class to feed the Tweets into the model
    """
    def __init__(self, tsv_address):
        self.tsv_address = tsv_address
        self.tweet_list = []
        self.current_index = 0

        self.loadTweets()  # Load tweets from the input CSV file
        self.size_ = len(self.tweet_list)  # Calculate the size of the tweet list

    def loadTweets(self):
        """
        Load tweets from the TSV file specified by input_csv_address.
        """
        df = pd.read_csv(self.tsv_address, delimiter='\t')
        # Only preserve the tweet column and maintain df as dataframe
        df = df[["tweet"]]
        df["true_misinfo"] = pd.Series(dtype='string')
        df["pred_misinfo"] = pd.Series(dtype='string')

        self.tweet_list = df.to_dict("records")

    def __iter__(self):
        """
        Allow the Tweetflow object to be iterated using a loop or other iterable constructs.
        """
        return iter(self.tweet_list)

    def size(self):
        """
        Return the size of the tweet list.
        """
        return self.size_
    