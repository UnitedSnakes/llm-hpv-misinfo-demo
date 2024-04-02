# Author: Shanglin Yang
# email: syang662@wisc.edu

""" This file generates and stores 2 * 6 templates. It also estimates the input token number and cost. Please check "TODO" before running this file.

All six input .csv files are stored under \template. Under that folder there is also a .csv that stores 12 generated templates.
"""

import os
import numpy as np
import pandas as pd
import tiktoken

CSV_NUMBERS = 6
# columns that we are interested in: tweet_id, title, valence_original
COL_NUMBER = 3
# we only take three rows as sample
ROW_NUMBER = 3

BASIC_PROMPT = ""

DETAILED_PROMPT = ""

# TODO
# might need to modify these
CSV_FOLDER_PATH = f"scripts{os.sep}Shanglin{os.sep}templates"
EST_OUTPUT_PATH = f"{CSV_FOLDER_PATH}{os.sep}estimation_results.txt"
TEMPLATE_OUTPUT_PATH = f"{CSV_FOLDER_PATH}{os.sep}processed_templates.csv"

# TODO
# https://openai.com/pricing
MODEL_NAME = "gpt-4-0125-preview"

# TODO
# https://openai.com/pricing
COST_PER_TOKEN = 1e-5  # input, gpt4


class TemplateGenerator:
    """Generate the total template and store in a csv file.
    """

    def __init__(
        self,
        csv_folder_path,
        csv_numbers,
        row_number,
        col_number,
        basic_prompt,
        detailed_prompt,
        output_path,
    ) -> None:
        """
        Args:
            csv_folder_path (str): where stores the input csv files
            csv_numbers (int): number of input csv files
            row_number (int): number of rows that will be used to generate the total template
            col_number (int): number of columns that will be used to generate the total template
            basic_prompt (str):
                basic prompt. First component of basic templates. More info at concat_df()'s comments
            detailed_prompt (str): detailed prompt. First component of detailed templates
            output_path (str): where to store the generated total template
        """
        self.csv_folder_path = csv_folder_path
        self.csv_numbers = csv_numbers
        self.row_number = row_number
        self.col_number = col_number
        self.basic_prompt = basic_prompt
        self.detailed_prompt = detailed_prompt
        self.output_path = output_path  # where to save the total template
        self.csv_index_dict = {}  # give each input csv an index
        self.longest_tweet = ""
        self.longest_tweet_len = 0
        self.longest_tweet_index = ["", -1]  # [filename, row index]

        # -------------------------------------
        # final df, expected size = 12, format:
        # {basic template from csv 1,
        #  detailed template from csv 1,
        #  ...,
        #  basic template from csv 6,
        #  detailed template from csv 6}
        # -------------------------------------
        self.df_total = pd.DataFrame(columns=["Template"])

    def find_longest_tweet(self):
        """Find the longest tweet among all csv files and update self.longest_tweet, self.longest_tweet_len and self.longest_tweet_index.
        """
        for filename in os.listdir(self.csv_folder_path):
            if filename.endswith("processed_templates.csv"):
                continue

            if filename.endswith(".csv"):
                csv_file_path = os.path.join(self.csv_folder_path, filename)
                df_csv = pd.read_csv(csv_file_path)
                df_tweets = df_csv["title"]  # "title" column

                for index, tweet in enumerate(df_tweets):
                    if len(tweet) > self.longest_tweet_len:
                        self.longest_tweet = tweet
                        self.longest_tweet_len = len(tweet)
                        self.longest_tweet_index = [
                            filename,
                            index + 1,
                        ]  # index starts from 1 here

    def concat_df(self):
        """Generate the total template by concatenating 12 templates.

        basic template = basic prompt + tweet1 + valence1 + t2 + v2 + t3 + v3
        detailed template = detailed prompt + tweet1 + valence1 + t2 + v2 + t3 + v3

        Remember to add these to both templates:
        1) Q: ... A: ...
        2) Tweet: <content>
        """

        file_count = 0  # check if exists correct number of csv files. Add 1 per iteration
        for filename in os.listdir(self.csv_folder_path):
            if filename.endswith("processed_templates.csv"):
                continue

            # append two templates (basic & detailed) from a csv in each iteration
            # 2 templates * 6 iter = 12 templates in total
            if filename.endswith(".csv"):
                self.csv_index_dict[file_count] = filename
                file_count += 1

                csv_file_path = os.path.join(self.csv_folder_path, filename)
                # A df that stores the whole csv
                df_csv = pd.read_csv(csv_file_path)

                title_col = df_csv["title"].head(3)  # size = 3 * 1
                valence_col = df_csv["valence_original"].head(3)  # size = 3 * 1

                # use lists to concatenate strings in loop. Will join these lists with "" later, separately
                basic_template_list = [f"{self.basic_prompt}\n\n"]
                detailed_template_list = [f"{self.detailed_prompt}\n\n"]

                for template_list in (basic_template_list, detailed_template_list):
                    for _ in range(self.row_number):  # 3
                        template_list.append(f"Q: Tweet: {title_col[_]}\n\n")
                        template_list.append(f"A: {valence_col[_]}\n\n")
                    template_list.append("Q: Tweet:<content>\n\nA:")

                basic_template = "".join(basic_template_list)
                detailed_template = "".join(detailed_template_list)

                self.df_total = pd.concat(
                    [self.df_total, pd.DataFrame({"Template": [basic_template]})],
                    ignore_index=True,
                )
                self.df_total = pd.concat(
                    [self.df_total, pd.DataFrame({"Template": [detailed_template]})],
                    ignore_index=True,
                )

        assert file_count == self.csv_numbers, "wrong numbers of csv files"
        assert len(self.df_total.shape) == 2 and self.df_total.shape == (
            self.csv_numbers * 2,
            1,
        ), f"df_total wrong size in TemplateGenerator"

    def save_as_csv(self):
        self.df_total.to_csv(self.output_path, index=False)
        print(f"Total template (2 * 6) has been stored in {self.output_path}\n")


class CostEstimator:
    """Estimate token number and cost for
    1) total template,
    2) template with most tokens,
    3) longest tweet among all csv files.
    Store results as a txt file.
    """

    def __init__(self, df_total, cost_per_token, model_name, output_path) -> None:
        """
        Args:
            df_total (df):
                the df storing the total template. Will be obtained from an instance of TemplateGenerator.
            cost_per_token (float): cost per token in dollar
            model_name (str): specifies a model version
            output_path (str): where to store the estimation results
        """
        self.df_total = df_total
        self.cost_per_token = cost_per_token
        self.model_name = model_name
        self.output_path = output_path

        # 1) total template
        self.num_tokens_template_total = 0
        self.cost_template_total = 0

        # 2) template with most tokens
        self.longest_template = ""
        self.num_tokens_longest_template = 0
        self.num_tokens_longest_template_filename = ""
        self.cost_longest_template = 0

        # 3) longest tweet among all csv files
        self.longest_tweet = ""
        self.num_tokens_longest_tweet = 0
        self.num_tokens_longest_tweet_index = ["", -1]  # [filename, row]
        self.cost_longest_tweet = 0

        assert 0 < cost_per_token < 3e-5, "wrong cost per token"

    def estimate(self, csv_index_dict, longest_tweet, longest_tweet_index):
        """Estimate tokens and cost for 1), 2) and 3).

        Args:
            csv_index_dict (dict):
                A dictionary storing indices for each input csv. Will be obtained from an instance of TemplateGenerator.
            longest_tweet (str):
                content of the longest tweet. Will be obtained from an instance of TemplateGenerator.
            longest_tweet_index (list):
                [filename, index] of the longest tweet. Will be obtained from an instance of TemplateGenerator.
        """
        encoding = tiktoken.encoding_for_model(self.model_name)

        # ------------------------------------------------------------
        # Example:
        # tokens = encoding.encode(template_total)
        # num_tokens = len(tokens)
        # self.cost_total_templates = num_tokens * self.cost_per_token
        # ------------------------------------------------------------

        # update tokens and cost for longest tweet among all csv files
        tokens = encoding.encode(longest_tweet)
        num_tokens = len(tokens)

        self.longest_tweet = longest_tweet[:20] + "..." + longest_tweet[-20:]
        self.num_tokens_longest_tweet = num_tokens
        self.num_tokens_longest_tweet_index = longest_tweet_index
        self.cost_longest_tweet = num_tokens * self.cost_per_token

        # update tokens and cost for the longest template among all 12 templates
        for index, template in enumerate(self.df_total["Template"]):
            tokens_template = encoding.encode(template)
            num_tokens_template = len(tokens_template)
            cost_template = num_tokens_template * self.cost_per_token

            if num_tokens_template > self.num_tokens_longest_template:
                self.longest_template = (
                    template[:10] + "..." + template[-75:-25] + "..." + template[-10:]
                ) # store a part of template rather than whole to save display space
                self.num_tokens_longest_template = num_tokens_template
                # remember // 2
                self.num_tokens_longest_template_filename = csv_index_dict[index // 2]
                self.cost_longest_template = cost_template

            # update total tokens and cost
            self.num_tokens_template_total += num_tokens_template
            self.cost_template_total += cost_template

    def save_estimation(self):
        # will store this dictionary
        pprint_dict = {
            "Cost per token in $": self.cost_per_token,
            "Model name": self.model_name,
            "All templates' tokens": self.num_tokens_template_total,
            "All templates' cost in $": self.cost_template_total,
            "Template with most tokens": self.longest_template,
            "Template with most tokens is generated from": self.num_tokens_longest_template_filename,
            "Token number in template with most tokens": self.num_tokens_longest_template,
            "Cost of template with most tokens in $": self.cost_longest_template,
            "Longest tweet": self.longest_tweet,
            "Longest tweet is generated from [filename, row index]": self.num_tokens_longest_tweet_index,
            "Token number in longest tweet": self.num_tokens_longest_tweet,
            "Cost of longest tweet in $": self.cost_longest_tweet,
        }

        with open(self.output_path, "w", encoding="utf-8") as file:
            printed_sep = False
            for k, v in pprint_dict.items():
                if (
                    k.startswith("All templates")
                    or k.startswith("Longest tweet")
                    or k.startswith("Template with")
                ):
                    if not printed_sep:
                        file.write("\n")
                        printed_sep = True
                    else:
                        printed_sep = False

                if isinstance(v, str):
                    v = v.replace("\n", "\\n")
                formatted_text = "{:<55}: {}".format(k, v)
                file.write(formatted_text + "\n")

        print(
            f"Estimation results have been saved in {self.output_path}\nPlease compare the results at https://platform.openai.com/tokenizer"
        )


if __name__ == "__main__":
    template_generator = TemplateGenerator(
        csv_folder_path=CSV_FOLDER_PATH,
        csv_numbers=CSV_NUMBERS,
        row_number=ROW_NUMBER,
        col_number=COL_NUMBER,
        basic_prompt=BASIC_PROMPT,
        detailed_prompt=DETAILED_PROMPT,
        output_path=TEMPLATE_OUTPUT_PATH,
    )

    template_generator.find_longest_tweet()
    template_generator.concat_df()
    template_generator.save_as_csv()

    cost_estimator = CostEstimator(
        df_total=template_generator.df_total,
        cost_per_token=COST_PER_TOKEN,
        model_name=MODEL_NAME,
        output_path=EST_OUTPUT_PATH,
    )

    cost_estimator.estimate(
        csv_index_dict=template_generator.csv_index_dict,
        longest_tweet=template_generator.longest_tweet,
        longest_tweet_index=template_generator.longest_tweet_index,
    )

    cost_estimator.save_estimation()
