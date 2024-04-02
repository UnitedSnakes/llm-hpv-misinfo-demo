# File: run_hpv.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import itertools
import os
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import pandas as pd
import pprint
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd
import openai
import re
import backoff
from joblib import Parallel, delayed
import torch as th

from tweetflow import Tweetflow
from parameters import *


# Author: Sean
# use backoff to handle the rate limit error
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


# Author: Sean
def predict_labels_one_core(df_input_text, prompt_template):
    """Label the stances of the texts with GPT-3.5-turbo-based models. This involves interacting with the model with OpenAI's API. This is the helper function that should be called by `predict_labels`. This helper function enables parallelization.

    Args:
        df_input_text (pd.DataFrame): the df that contains the texts to be labeled.
        prompt_template (str)L: the prompt template where the tweet will be embedded in.

    Returns:
        list_tweet_embedded (list): the list of embedded tweets.
        list_tweet_id (list): the list of tweet ids.
        list_output_labels (list): the list of predicted labels.
        list_tweet_warnings (list): the list of tweets that have warnings.
        list_message_warnings (list): the list of warning messages.
    """
    # --------------------
    # predict the labels
    # --------------------
    # - collect the inputs and results
    list_tweet_embedded = []
    list_tweet_id = []
    list_output_labels = []
    # - for warning messages only
    list_tweet_warnings = []
    list_message_warnings = []
    try:
        prompts = (
            df_input_text["tweet"]
            .apply(lambda x: prompt_template.format(tweet_content=x))
            .tolist()
        )

        for i_tweet, (prompt, tweet_id) in enumerate(
            zip(
                prompts,
                df_input_text["tweet_id"].values,
            )
        ):
            print("tweet: {}/{}".format(i_tweet, len(df_input_text)))
            try:
                chat = completions_with_backoff(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
            except Exception as e:
                message_warning = "Tweet ID: {}. Error: {} ".format(tweet_id, str(e))
                print(message_warning)
                list_tweet_warnings.append(tweet_id)
                list_message_warnings.append(message_warning)
                time.sleep(30)
                chat = completions_with_backoff(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
            # --------------------
            # Parse the output and collect the results
            # --------------------
            output_label = chat.choices[0].message.content
            if chat.choices[0].finish_reason != "stop":
                message_warning = (
                    "Tweet ID: {}. Warning: the prediction is not finished. ".format(
                        tweet_id
                    )
                )
                print(message_warning)
                list_tweet_warnings.append(tweet_id)
                list_message_warnings.append(message_warning)
            list_tweet_embedded.append(prompt)
            list_tweet_id.append(tweet_id)
            list_output_labels.append(output_label)
    except Exception as e:
        print("Not all tweets are predicted. Error: {}".format(str(e)))
        print("Save the predictions so far.")
    return (
        list_tweet_embedded,
        list_tweet_id,
        list_output_labels,
        list_tweet_warnings,
        list_message_warnings,
    )


def run_gpt_model(df_prompts, prompt_template):
    # parallelize
    if NUM_CORES_CALL_API > 1:
        # - split the input df into chunks
        num_cores = NUM_CORES_CALL_API
        list_df_input_text = np.array_split(df_prompts, num_cores)

        # - use Parallel and delayed to execute the function in parallel:
        list_results = Parallel(n_jobs=num_cores)(
            delayed(predict_labels_one_core)(df_input_text_chunk, prompt_template)
            for df_input_text_chunk in list_df_input_text
        )
        (
            list_tweet_embedded,
            list_tweet_id,
            list_output_labels,
            list_tweet_warnings,
            list_message_warnings,
        ) = (list(itertools.chain(*result_list)) for result_list in zip(*list_results))

        return list_results

    elif NUM_CORES_CALL_API == 1:
        # # Constrcut a batch request
        # batch_requests = []

        batch_responses = []
        count = 0
        for prompt in prompts:
            # request = {
            #     "engine": MODEL,  # Specify engine to be "gpt_chat_4", "gpt_chat_turbo_3_5",
            #     # "gpt_3_davinci", or "gpt_2"
            #     "prompt": prompt,
            #     "max_tokens": 10,
            #     "temperature": 0.7,
            # }
            # batch_requests.append(request)

            # messages = [{"role": "user", "content": json.dumps(prompts)}]

            # batch_instruction = {
            #     "role":
            #     "system",
            #     "content":
            #     "Complete every element of the array. Reply with an array of all completions."
            # }

            # messages.append(batch_instruction)

            # print(messages)
            # # Send the API request to OPENAI
            # batch_responses = openai.ChatCompletion.create(
            #     model=MODEL, messages=batch_requests, temperature=0.7, max_tokens=10 * len(batch_requests)
            # )

            # print("API request has been sent.")

            # for message in messages:

            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            count += 1
            print(f"Running prompt {count}...")

            batch_responses.append(
                openai.ChatCompletion.create(
                    model=MODEL,
                    messages=message,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
            )
        # print("API request has been sent.")
        # print(batch_responses.choices)
        completions = [
            response.choices[0].message.content for response in batch_responses
        ]
        # print(completions)
        # Process each completion
        # for i, response in enumerate(batch_responses.choices):
        #     completions.append(response.message["content"])

        return completions
    else:
        raise NotImplementedError


def run_flan_model(device, row_batch):
    """
    Generates predictions for the given input texts.

    Args:
        prompts (list): A list of input texts to generate predictions for.

    Returns:
        list: A list of generated texts as predictions.
    """
    if MODEL == "flan-ul2":
        quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True, llm_int8_threshold=5.0)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
        # device_map = {"shared": 0, "encoder": 0, "decoder": 1, "lm_head": 0}
        # device_map = {'shared': 0, 'decoder.embed_tokens': 0, 'encoder': 0, 'lm_head': 'cpu', 'decoder.block.0': 0, 'decoder.block.1': 0, 'decoder.block.2': 0, 'decoder.block.3': 1, 'decoder.block.4': 1, 'decoder.block.5': 1, 'decoder.block.6': 1, 'decoder.block.7': 1, 'decoder.block.8': 1, 'decoder.block.9': 1, 'decoder.block.10': 1, 'decoder.block.11': 1, 'decoder.block.12': 1, 'decoder.block.13': 1, 'decoder.block.14': 1, 'decoder.block.15': 1, 'decoder.block.16': 1, 'decoder.block.17': 1, 'decoder.block.18': 1, 'decoder.block.19': 1, 'decoder.block.20': 1, 'decoder.block.21': 1, 'decoder.block.22': 1, 'decoder.block.23': 1, 'decoder.block.24': 1, 'decoder.block.25': 1, 'decoder.block.26': 1, 'decoder.block.27': 1, 'decoder.block.28': 1, 'decoder.block.29': 1, 'decoder.block.30': 1, 'decoder.block.31': 1, 'decoder.final_layer_norm': 1, 'decoder.dropout': 1}
        model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-ul2", device_map="auto", load_in_8bit=True, quantization_config=quantization_config, torch_dtype=th.bfloat16
        )
        # model = T5ForConditionalGeneration.from_pretrained(
        #     "google/flan-ul2", device_map="auto"
        # )
        # print(model.hf_device_map)
    else:
        tokenizer = T5Tokenizer.from_pretrained("google/" + MODEL)
        model = T5ForConditionalGeneration.from_pretrained(
            "google/" + MODEL, device_map="auto", torch_dtype=th.bfloat16
        )
    # model.parallelize()

    error_msg = ""
    exception_csv_msg = pd.DataFrame(columns=["tweet_id", "prompt", "reason", "label"])

    # try:
    qualified_prompts = []
    unqualified_indices = []

    # filter out the prompt(s) exceeding the max length
    for i, row in enumerate(row_batch.itertuples(), start=row_batch.index[0]):
        # len_tokens = tokenizer(row.prompt, return_tensors="pt", padding=True)[
        #     "input_ids"
        # ].shape[1]

        # if len_tokens > MAX_INPUT_TOKENS:
        #     unqualified_indices.append(i)

        #     error_msg_with_prompt = f"Prompt with tweet_id = {row.tweet_id} exceeding maximal input token limit: {len_tokens} > {MAX_INPUT_TOKENS}. Prompt with tweet_id = {row.tweet_id}:\n\n{row.prompt}"
        #     print(error_msg_with_prompt)
        #     error_msg += error_msg_with_prompt

        #     exception_csv_msg.loc[len(exception_csv_msg.index)] = [
        #         row.tweet_id,
        #         row.prompt,
        #         f"Exceeding maximal input token limit: {len_tokens} > {MAX_INPUT_TOKENS}.",
        #         row.true_misinfo,
        #     ]

        # else:
        #     qualified_prompts.append(row.prompt)
        qualified_prompts.append(row.prompt)

    input_ids = tokenizer(
        qualified_prompts, return_tensors="pt", padding=True
    ).input_ids.to(device)

    # print()
    # for i in qualified_prompts:
    #     print(i + "\n")
    # print()

    print("----------------------------------------------------------------")
    print("input_ids:")
    print("----------------------------------------------------------------")
    for i in range(len(input_ids)):
        print(f"{i + 1}")
        print(tokenizer.decode(input_ids[i], skip_special_tokens=True))
        print()
        print()
        print()

    outputs = model.generate(input_ids, max_new_tokens=10)
    output_labels = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # delete those tweets exceeding token limit from row_batch
    row_batch.drop(unqualified_indices, inplace=True)

    row_batch["pred_misinfo"] = output_labels

    # except Exception as e:
    #     print(e)
    #     error_msg += f"\n\n{e}\n"

    return row_batch, error_msg, exception_csv_msg


def predict(prompt_version, prompt_template):
    df_prompt_version = pd.DataFrame(
        columns=["tweet_id", "prompt", "true_misinfo", "pred_misinfo"]
    )
    delta_index = 0

    device = None
    if (
        MODEL[:4] == "flan"
    ):  # Try to use cuda to speed up the computation when using flan
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        print(f"Device: {device}")

    error_messages = []
    exception_csv_messages = pd.DataFrame(
        columns=["tweet_id", "prompt", "reason", "label"]
    )

    # for input_csv in ["test_positive_10.csv", "test_negative_10.csv"]:
    for input_csv in ["misinfo_tweets.csv", "non_misinfo_tweets.csv"]:
        input_csv_path = os.path.join(DATASET_MISINFO_OR_NOT_FOLDER_PATH, input_csv)

        if not os.path.isfile(input_csv_path):
            raise FileNotFoundError(f"{input_csv_path} does not exist!")

        tweetflow = Tweetflow(
            input_csv_path
        )  # columns are [tweet, true_misinfo, pred_misinfo]

        tweetflow_df = pd.DataFrame(tweetflow)
        tweetflow_df.index += 1

        # update tweet_id column
        tweetflow_df.index += delta_index
        tweetflow_df = tweetflow_df.reset_index().rename(columns={"index": "tweet_id"})
        delta_index = tweetflow_df.index.max() + 1

        if prompt_version != "zero":
            # Delete tweets that were used as few-shot examples
            mask = tweetflow_df["tweet"].isin(FEW_SHOT_EXAMPLES_DICT[prompt_version])
            tweetflow_df = tweetflow_df[~mask]

        # tweetflow_df["true_misinfo"] = (
        #     "yes" if input_csv == "test_positive_10.csv" else "no"
        # )
        tweetflow_df["true_misinfo"] = (
            "yes" if input_csv == "misinfo_tweets.csv" else "no"
        )

        print(
            f"Start processing Tweets in {input_csv} with {MODEL} and {prompt_version}.\n"
        )

        if MODEL[:4] == "flan":
            # divide the tweets into batches
            # the last batch might be smaller than batch_size
            for batch_count, group_index in enumerate(
                range(0, len(tweetflow_df), BATCH_SIZE), start=1
            ):
                row_batch = tweetflow_df.iloc[group_index : group_index + BATCH_SIZE]

                print(f"Processing batch {batch_count}...")

                # embed the tweets
                row_batch.loc[:, "tweet"] = row_batch["tweet"].apply(
                    lambda x: prompt_template.format(tweet_content=x)
                )
                row_batch = row_batch.rename(columns={"tweet": "prompt"})

                row_batch, error_msg, exception_csv_msg = run_flan_model(
                    device, row_batch
                )

                if row_batch["pred_misinfo"].isnull().any():
                    print(f"Empty response at {input_csv_path}!")
                    error_messages.append(
                        f"\n\n{prompt_version} with {input_csv} and {MODEL} in batch #{batch_count}: Empty response!"
                    )

                if error_msg:
                    error_messages.append(
                        f"{prompt_version} with {input_csv} and {MODEL} in batch #{batch_count}:\n\n{error_msg}"
                    )

                if not exception_csv_msg.empty:
                    exception_csv_messages = pd.concat(
                        [exception_csv_messages, exception_csv_msg]
                    )

                print(
                    f"Accumulated error_messages after {prompt_version} with {input_csv} and {MODEL} in batch #{batch_count}:\n\n{error_messages}"
                )
                print(row_batch["pred_misinfo"])

                # regularize the completions to be either "yes" or "no"
                pred_misinfo_regularized = []

                for completion in row_batch["pred_misinfo"]:
                    matches = re.findall(PATTERN, completion.lower())
                    if len(matches) != 1:
                        raise ValueError(
                            f"Completion does not contain exactly one of 'yes' or 'no'.\nCompletion: {completion}"
                        )

                    pred_misinfo_regularized.append(matches[0])

                row_batch.loc[:, "pred_misinfo"] = pred_misinfo_regularized

                # update df by concatenating the new sub-df to it
                df_prompt_version = pd.concat([df_prompt_version, row_batch])

        elif MODEL[:3] == "gpt":
            # GPT family

            list_results = run_gpt_model(tweetflow_df, prompt_template)

            completions_with_tweet_id = [
                list_result[1:3] for list_result in list_results
            ]
            completions_with_tweet_id = [
                list(itertools.chain(*[_[0] for _ in completions_with_tweet_id])),
                list(itertools.chain(*[_[1] for _ in completions_with_tweet_id])),
            ]

            if completions_with_tweet_id[0] != sorted(completions_with_tweet_id[0]):
                print("Unsroted completions in parallel. Sorted now.")
                combined_data = list(zip(*completions_with_tweet_id))

                sorted_data = sorted(combined_data, key=lambda x: x[0])
                completions_with_tweet_id = list(zip(*sorted_data))

            pred_misinfo = []

            for completion in completions_with_tweet_id[1]:
                matches = re.findall(PATTERN, completion.lower())
                if len(matches) != 1:
                    print(
                        f"Completion does not contain exactly one of 'yes' or 'no'.\nCompletion: {completion}"
                    )

                pred_misinfo.append(matches[0])

            tweetflow_df["pred_misinfo"] = pred_misinfo

            df_prompt_version = pd.concat([df_prompt_version, tweetflow_df])

        print(f"\nFinished processing {input_csv} with {MODEL} and {prompt_version}.\n")

    df_prompt_version.index += 1

    df_prompt_version.to_csv(
        f"{os.path.join(output_path_model_prompt, prompt_version)}_labels.csv",
        index=False,
    )

    post_process_data(df_prompt_version, prompt_version)

    if not exception_csv_messages.empty:
        exception_csv_messages.to_csv(
            f"{os.path.join(OUTPUT_MISINFO_OR_NOT_FOLDER_PATH, MODEL, prompt_version, prompt_version)}_exceptions.csv",
            index=False,
        )

    return df_prompt_version, error_messages, device


def post_process_data(df, prompt_version):
    global summary_total

    true_labels = df["true_misinfo"]
    predicted_labels = df["pred_misinfo"]

    f1_macro = f1_score(true_labels, predicted_labels, average="macro")
    f1_yes = f1_score(true_labels, predicted_labels, labels=["yes"], average=None)[0]
    f1_no = f1_score(true_labels, predicted_labels, labels=["no"], average=None)[0]

    recall_macro = recall_score(true_labels, predicted_labels, average="macro")
    recall_yes = recall_score(
        true_labels, predicted_labels, labels=["yes"], average=None
    )[0]
    recall_no = recall_score(
        true_labels, predicted_labels, labels=["no"], average=None
    )[0]

    precision_macro = precision_score(true_labels, predicted_labels, average="macro")
    precision_yes = precision_score(
        true_labels, predicted_labels, labels=["yes"], average=None
    )[0]
    precision_no = precision_score(
        true_labels, predicted_labels, labels=["no"], average=None
    )[0]

    print(f1_macro, precision_macro, recall_macro)

    summary_data = {
        "f1_macro": [f1_macro],
        "f1_yes": [f1_yes],
        "f1_no": [f1_no],
        "recall_macro": [recall_macro],
        "recall_yes": [recall_yes],
        "recall_no": [recall_no],
        "precision_macro": [precision_macro],
        "precision_yes": [precision_yes],
        "precision_no": [precision_no],
    }

    summary = pd.DataFrame(summary_data)

    print(summary)
    summary.index = [prompt_version]
    summary_total = pd.concat([summary_total, summary])

    # Retrieve confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)

    # Rename rows and columns
    confusion = pd.DataFrame(
        confusion,
        columns=["predicted_non_misinfo", "predicted_misinfo"],
        index=["true_non_misinfo", "true_misinfo"],
    )

    print(confusion)

    print(
        f"{os.path.join(OUTPUT_MISINFO_OR_NOT_FOLDER_PATH, MODEL, prompt_version, prompt_version)}_confusion_matrix.csv"
    )

    # write into two .csv files
    confusion.to_csv(
        f"{os.path.join(OUTPUT_MISINFO_OR_NOT_FOLDER_PATH, MODEL, prompt_version, prompt_version)}_confusion_matrix.csv"
    )

    # Quick review
    print(
        f"Summary for {os.path.join(OUTPUT_MISINFO_OR_NOT_FOLDER_PATH, MODEL, prompt_version, prompt_version)}_labels.csv:\n\n"
    )
    pprint.pprint(summary)


def post_process_data_total():
    # rename the rows
    global summary_total, df_all_prompt_versions

    # expected_keys = {0: "zero"}
    expected_keys = {0: "zero", 1: "few_k2", 2: "few_k4", 3: "few_k6"}
    missing_prompt_versions = [
        prompt_version
        for index, prompt_version in expected_keys.items()
        if prompt_version not in df_all_prompt_versions
    ]

    for missing_prompt_version in missing_prompt_versions:
        post_process_data(
            pd.read_csv(
                f"{os.path.join(OUTPUT_MISINFO_OR_NOT_FOLDER_PATH, MODEL, missing_prompt_version, missing_prompt_version)}_labels.csv"
            ),
            missing_prompt_version,
        )

    summary_total = summary_total.reindex(list(expected_keys.values()))
    print(summary_total)

    summary_total.to_csv(
        f"{os.path.join(OUTPUT_MISINFO_OR_NOT_FOLDER_PATH, MODEL) + os.sep}classification_summary.csv"
    )


def record_in_context_prompt_examples(prompt_version):
    in_context_examples = []

    if prompt_version == "zero":
        in_context_examples = [["None", "None"]]
    elif prompt_version == "few_k2":
        in_context_examples = zip(
            FEW_SHOT_EXAMPLES[:2],
            ["misinfo_strong_cases (yes)", "all_negative_cases (no)"],
        )
    elif prompt_version == "few_k4":
        in_context_examples = zip(
            FEW_SHOT_EXAMPLES[:4],
            [
                "misinfo_strong_cases (yes)",
                "all_negative_cases (no)",
                "misinfo_strong_cases (yes)",
                "non_negative_cases (no)",
            ],
        )
    elif prompt_version == "few_k6":
        in_context_examples = zip(
            FEW_SHOT_EXAMPLES[:],
            [
                "misinfo_strong_cases (yes)",
                "all_negative_cases (no)",
                "misinfo_strong_cases (yes)",
                "non_negative_cases (no)",
                "misinfo_strong_cases (yes)",
                "skepticism (no)",
            ],
        )
    else:
        raise NotImplementedError(f"Prompt version {prompt_version} not recognized.")

    in_context_examples_df = pd.DataFrame(
        in_context_examples, columns=["examples", "description"]
    )

    in_context_examples_df.index += 1

    in_context_examples_df.to_csv(
        f"{os.path.join(output_path_model_prompt, prompt_version)}_in_context_prompt_examples.csv"
    )


if __name__ == "__main__":
    """
    Runs the prediction process.
    Feed the model in batches to speed up the prediction process.
    """
    t1 = time.time()

    summary_total = pd.DataFrame(
        columns=[
            "f1_macro",
            "f1_yes",
            "f1_no",
            "recall_macro",
            "recall_yes",
            "recall_no",
            "precision_macro",
            "precision_yes",
            "precision_no",
        ]
    )

    df_all_prompt_versions = {}
    error_messages_total = []

    device = "N/A"

    for prompt_version, prompt_template in PROMPT_DICT.items():
        # if prompt_version != "zero":
        #     continue
        output_path_model_prompt = os.path.join(
            OUTPUT_MISINFO_OR_NOT_FOLDER_PATH, MODEL, prompt_version
        )

        if os.path.exists(
            f"{os.path.join(output_path_model_prompt, prompt_version)}_labels.csv"
        ):
            print(
                f"{os.path.join(output_path_model_prompt, prompt_version)}_labels.csv already exists.\n"
            )
            skip = False
            while True:
                user_input = input("Overwrite? [Y/n]: ")
                if user_input.lower() == "y":
                    print("Overwriting...")
                    break
                elif user_input.lower() == "n":
                    print(
                        f"{os.path.join(output_path_model_prompt, prompt_version)}_labels.csv is skipped."
                    )
                    skip = True
                    break
                else:
                    print("Invalid input.")
            if skip:
                continue

        if not os.path.exists(output_path_model_prompt):
            os.makedirs(output_path_model_prompt)
        record_in_context_prompt_examples(prompt_version)

        df_prompt_version, error_messages, device = predict(
            prompt_version, prompt_template
        )

        if error_messages:  # error messages is prompt-wise
            error_messages_total.append(error_messages)

        df_all_prompt_versions[prompt_version] = df_prompt_version
    post_process_data_total()

    t2 = time.time()
    print("\n-------------------error messages-------------------")
    print(f"{error_messages_total}\n\n")

    execution_summary = {
        "device": str(device),
        "batch size": BATCH_SIZE,
        "time": t2 - t1,
    }

    print("-----------------execution summary-----------------\n\n")
    pprint.pprint(execution_summary)
