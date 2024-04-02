# File: processor_misinfo_or_not.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import os
import re
import pandas as pd
from parameters import (
    DATASET_ORIGINAL_FOLDER_PATH,
    DATASET_MISINFO_OR_NOT_FOLDER_PATH,
    OUTPUT_MISINFO_OR_NOT_FOLDER_PATH,
)


# Expected:
# Sample_tweets_for_GT - all_negative_cases.tsv = 417 -> 419 tweets
# Sample_tweets_for_GT - misinfo_non_strong.tsv = 145 -> 144 tweets
# Sample_tweets_for_GT - misinfo_strong_cases.tsv = 60 tweets
# Sample_tweets_for_GT - non_negative_cases.tsv = 601 tweets
# Sample_tweets_for_GT - skepticism.tsv = 73 tweets
# misinfo_tweets.tsv = 201 tweets
# non_misinfo_tweets.tsv = 868 tweets


def remove_specific_suffix(input_string, suffixes):
    input_string = input_string.strip()
    pattern = r'^"(.*)"$'
    input_string = re.sub(pattern, r'\1', input_string)
    for suffix in suffixes:
        pattern = re.escape(suffix) + r"$"
        input_string = re.sub(pattern, "", input_string.strip())
    return input_string.strip()


if __name__ == "__main__":
    for output_file in [
        f"{DATASET_MISINFO_OR_NOT_FOLDER_PATH + os.sep}misinfo_tweets.csv",
        f"{DATASET_MISINFO_OR_NOT_FOLDER_PATH + os.sep}non_misinfo_tweets.csv",
    ]:
        if os.path.exists(output_file):
            print(f"\n{output_file} already exists.\n")
            user_input = input("Overwrite? [Y/n]: ")

            if user_input.lower() != "y":
                print(f"{output_file} is skipped.")
                exit()

            print("Overwriting...")

    suffixes = [
        "[Any negative or positive content counts]",
        "- Q1. What is the tweet‚Äôs valence toward HPV vaccines?",
        "2-Negative",
        "- Q2. If the tweet has negative information about HPV vaccines, does the tweet contain any misinformation?",
    ]
    
    
    # Manually update Sample_tweets_for_GT - all_negative_cases.tsv as Luhang suggested
    df = pd.read_csv(f"{DATASET_ORIGINAL_FOLDER_PATH + os.sep}Sample_tweets_for_GT - all_negative_cases.tsv", sep='\t')
    df = df.iloc[:, 1:]
    df.index += 1
    df.insert(0, 'number', df.index)
    df.to_csv(f"{DATASET_ORIGINAL_FOLDER_PATH + os.sep}Sample_tweets_for_GT - all_negative_cases.tsv", sep='\t', index=False)
    
    # Manually update Sample_tweets_for_GT - misinfo_non_strong.tsv as Luhang suggested
    df = pd.read_csv(f"{DATASET_ORIGINAL_FOLDER_PATH + os.sep}Sample_tweets_for_GT - misinfo_non_strong.tsv", sep='\t')
    df = df.iloc[:, 1:]
    df.index += 1
    df.insert(0, 'number', df.index)
    df.to_csv(f"{DATASET_ORIGINAL_FOLDER_PATH + os.sep}Sample_tweets_for_GT - misinfo_non_strong.tsv", sep='\t', index=False)
    
    with open(
        f"{DATASET_ORIGINAL_FOLDER_PATH + os.sep}Sample_tweets_for_GT - all_negative_cases.tsv",
        "r",
    ) as file_a, open(
        f"{DATASET_ORIGINAL_FOLDER_PATH + os.sep}Sample_tweets_for_GT - misinfo_non_strong.tsv",
        "r",
    ) as file_b, open(
        f"{DATASET_ORIGINAL_FOLDER_PATH + os.sep}Sample_tweets_for_GT - misinfo_strong_cases.tsv",
        "r",
    ) as file_c, open(
        f"{DATASET_ORIGINAL_FOLDER_PATH + os.sep}Sample_tweets_for_GT - non_negative_cases.tsv",
        "r",
    ) as file_d, open(
        f"{DATASET_ORIGINAL_FOLDER_PATH + os.sep}Sample_tweets_for_GT - skepticism.tsv",
        "r",
    ) as file_e:
        misinfo_tweets = set()
        non_misinfo_tweets = set()

        misinfo_non_strong_tweets = set()
        # skip the row of column names
        next(file_b, None)
        # check duplicates
        unique_tweets = set()
        count = 0
        for line in file_b:
            tweet = line.strip().split("\t")[2]
            tweet = remove_specific_suffix(tweet, suffixes)
            if tweet not in unique_tweets:
                unique_tweets.add(tweet)
            else:
                count += 1
                print(f"Number {count} duplicate tweet in Sample_tweets_for_GT - misinfo_non_strong.tsv:\n\n{tweet}\n")
            misinfo_tweets.add(tweet)
            misinfo_non_strong_tweets.add(tweet)

        misinfo_strong_cases_tweets = set()
        # skip the row of column names
        next(file_c, None)
        # check duplicates
        unique_tweets = set()
        count = 0
        for line in file_c:
            tweet = line.strip().split("\t")[2]
            tweet = remove_specific_suffix(tweet, suffixes)
            if tweet not in unique_tweets:
                unique_tweets.add(tweet)
            else:
                count += 1
                print(f"Number {count} duplicate tweet in Sample_tweets_for_GT - misinfo_strong_cases.tsv:\n\n{tweet}\n")
            misinfo_tweets.add(tweet)
            misinfo_strong_cases_tweets.add(tweet)

        non_negative_tweets = set()
        # skip the row of column names
        next(file_d, None)
        # check duplicates
        unique_tweets = set()
        count = 0
        for line in file_d:
            tweet = line.strip().split("\t")[1]
            tweet = remove_specific_suffix(tweet, suffixes)
            if tweet not in unique_tweets:
                unique_tweets.add(tweet)
            else:
                count += 1
                print(f"Number {count} duplicate tweet in Sample_tweets_for_GT - non_negative_cases.tsv:\n\n{tweet}\n")
            non_misinfo_tweets.add(tweet)
            non_negative_tweets.add(tweet)

        skepticism_tweets = set()
        # skip the row of column names
        next(file_e, None)
        # check duplicates
        unique_tweets = set()
        count = 0
        for line in file_e:
            tweet = line.strip().split("\t")[1]
            tweet = remove_specific_suffix(tweet, suffixes)
            if tweet not in unique_tweets:
                unique_tweets.add(tweet)
            else:
                count += 1
                print(f"Number {count} duplicate tweet in Sample_tweets_for_GT - skepticism.tsv:\n\n{tweet}\n")
            non_misinfo_tweets.add(tweet)
            skepticism_tweets.add(tweet)

        all_neg_tweets = set()
        # skip the row of column names
        next(file_a, None)
        # check duplicates
        unique_tweets = set()
        count = 0
        for line in file_a:
            tweet = line.strip().split("\t")[1]
            tweet = remove_specific_suffix(tweet, suffixes)
            if tweet not in unique_tweets:
                unique_tweets.add(tweet)
            else:
                count += 1
                print(f"Number {count} duplicate tweet in Sample_tweets_for_GT - all_negative_cases.tsv:\n\n{tweet}\n")
            all_neg_tweets.add(tweet)
            if tweet not in misinfo_tweets:
                non_misinfo_tweets.add(tweet)

    sets = [
        misinfo_tweets,  # 0
        non_misinfo_tweets,
        misinfo_non_strong_tweets,  # 2
        misinfo_strong_cases_tweets,
        non_negative_tweets,  # 4
        skepticism_tweets,
        all_neg_tweets,  # 6
    ]

    with open(
        f"{DATASET_MISINFO_OR_NOT_FOLDER_PATH + os.sep}misinfo_tweets.csv", "w"
    ) as file_f:
        file_f.write("tweet\n")
        for tweet in misinfo_tweets:
            file_f.write(f"{tweet}\n")

    with open(
        f"{DATASET_MISINFO_OR_NOT_FOLDER_PATH + os.sep}non_misinfo_tweets.csv", "w"
    ) as file_g:
        file_g.write("tweet\n")
        for tweet in non_misinfo_tweets:
            file_g.write(f"{tweet}\n")

    print(
        f"Two files have been generated:\n{OUTPUT_MISINFO_OR_NOT_FOLDER_PATH + os.sep}misinfo_tweets.csv\n{OUTPUT_MISINFO_OR_NOT_FOLDER_PATH + os.sep}non_misinfo_tweets.csv"
    )

    # intersection = sets[2].intersection(sets[3])
    # intersection.remove('tweet')
    # print(intersection)

    # print(len(sets[2])) # 144 but 143?
    # print(len(sets[3])) # 60
    # print(len(sets[0])) # 202?
    # print(sets[0] == sets[2] | sets[3]) # True
    # for i in list(sets[3]):
        # print(i)

    # # Intersection matrix
    # matrix = [[0] * 7 for _ in range(7)]

    # for i in range(7):
    #     for j in range(i, 7):
    #         if i == j:
    #             matrix[i][j] = 1
    #         else:
    #             intersection = sets[i].intersection(sets[j])
    #             intersection.remove('tweet')
    #             if len(intersection) > 0:
    #                 matrix[i][j] = matrix[j][i] = 1

    # for row in matrix:
    #     print(row)
