import ast
import warnings
import pandas as pd
import re
import numpy as np
import os
from config import Config


# Define ANSI escape sequences for colored terminal text
class Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def request_user_confirmation(
    question_prompt, yes_prompt, no_prompt, auto_confirm=None
):
    """
    Prompts the user for confirmation with a yes/no question.

    Args:
        question_prompt (str): The question to ask the user.
        yes_prompt (str): The message to display if the user confirms.
        no_prompt (str): The message to display if the user declines.
        auto_confirm (str, optional): Automatically confirm based on its value ('y' or 'n'). Defaults to None.

    Returns:
        bool: True if the user confirms, False otherwise.
    """
    if auto_confirm is not None:
        if auto_confirm.lower() == "y":
            print(yes_prompt)
            return True
        elif auto_confirm.lower() == "n":
            print(no_prompt)
            return False

    while True:
        user_input = input(question_prompt)
        if user_input.lower() == "y":
            print(yes_prompt)
            return True
        elif user_input.lower() == "n":
            print(no_prompt)
            return False
        else:
            print("Invalid input. Please try again.")


def find_label(completion):
    """
    Find the first keyword in a completion from a list of keywords.

    Args:
        completion (str): The text to search within.
        keywords (list): A list of keywords to search for.

    Returns:
        str: The first keyword found or None if no keywords are found.
    """
    completion_lower = completion.lower().strip()

    label = None
    label_status = None

    if completion_lower.startswith(("not inferred", "not inferenced")):
        return "not_inferred", "not_inferred"

    prefixes = [
        '"',
        ":",
        "|",
        "-",
        "{",
        "(",
        "**",
        "a:",
        "b:",
        "c:",
        "a.",
        "b.",
        "c.",
        "a;",
        "b;",
        "c;",
        "a>",
        "b>",
        "c>",
        "1.",
        "2.",
        "3.",
        "1)",
        "2)",
        "3)",
        "assistant",
        "assistant:",
    ]

    for _ in range(3):
        for prefix in prefixes:
            if completion_lower.startswith(prefix):
                completion_lower = completion_lower[len(prefix) :]

        completion_lower = completion_lower.strip()

    break_flag = False
    for synonyms in Config.patterns:
        for pattern in synonyms:
            if completion_lower.startswith(pattern):
                label = synonyms[0]
                completion_lower = completion_lower.removeprefix(pattern)
                break_flag = True
                break
        if break_flag:
            break

    # keyword_positions = [
    #     (keyword, completion_lower.find(keyword)) for keyword in keywords
    # ]
    # # Remove keywords not found in the text
    # keyword_positions = [
    #     (kw, pos) for kw, pos in keyword_positions if pos != -1
    # ]

    immediate = True if label else False
    multiple = False
    break_flag = False

    for synonyms in Config.patterns:
        if label == synonyms[0]:
            continue
        for pattern in synonyms:
            if completion_lower.find(pattern) != -1:
                if immediate or (label and not immediate):
                    multiple = True  # immediate multiple or nonimmediate multiple
                    break_flag = True
                    if not immediate:
                        label = None
                    break
                label = synonyms[0]  # nonimmediate, might be single or multiple
                break
        # immediate single or no label
        if break_flag:
            break

    if immediate:
        label_status = "immediate_single" if not multiple else "immediate_multiple"
        return label, label_status

    # Nonimmediate label
    if multiple:
        return label, "nonimmediate_multiple"
    elif label:
        return label, "nonimmediate_single"
    return label, "no_label"


# WARNING: This tweet needs manual inspection: id=15372, tweet_id=250315297244, label_status=nonimmediate_multiple, completion=The stance of the tweet with respect to vaccination against human papilloma virus (HPV) is "in favor."
def check_completion_format(df, extract=False):
    """
    Check if the completion format in a DataFrame matches the expected formats.

    Args:
        df (pd.DataFrame): DataFrame containing the completions to check.
        extract (bool, optional): If True, attempts to extract and correct the label. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing warnings for completions that do not match the expected formats.
    """
    df_warnings = pd.DataFrame(columns=["id", "tweet_id", "inference_results", "level"])

    immediate_single = 0  # completion starts with a keyword, and there is only one keyword in the completion
    immediate_multiple = 0  # completion starts with a keyword, and there are multiple keywords in the completion
    nonimmediate_single = 0  # completion does not start with a keyword, and there is only one keyword in the completion
    nonimmediate_multiple = 0  # completion does not start with a keyword, and there are multiple keywords in the completion
    no_label = 0  # completion does not contain any of the keywords
    not_inferred = 0

    for index, (id, tweet_id, completion) in enumerate(
        zip(df["id"], df["tweet_id"], df["inference_results"])
    ):
        try:
            # if id == 15372:
            #     print(111)
            label, label_status = find_label(completion)

            if extract and label:
                df.loc[index, "inference_results"] = label

            need_manual = False

            if label_status == "immediate_single":
                immediate_single += 1
            elif label_status == "immediate_multiple":
                immediate_multiple += 1
            elif label_status == "nonimmediate_single":
                nonimmediate_single += 1
            elif label_status == "nonimmediate_multiple":
                nonimmediate_multiple += 1
                need_manual = True
            elif label_status == "no_label":
                no_label += 1
                need_manual = True
            else:
                not_inferred += 1

            if need_manual:
                new_row = pd.DataFrame(
                    [
                        {
                            "id": id,
                            "tweet_id": tweet_id,
                            "inference_results": completion,
                            "level": 1,
                        }
                    ]
                )
                df_warnings = pd.concat([df_warnings, new_row], ignore_index=True)
                print(
                    f"WARNING: This tweet needs manual inspection: id={id}, tweet_id={tweet_id}, label_status={label_status}, completion={completion}"
                )

        except:
            print(
                f"WARNING: This tweet obtained erroneous inference: id={id}, tweet_id={tweet_id}, completion={completion}"
            )
            new_row = pd.DataFrame(
                [
                    {
                        "id": id,
                        "tweet_id": tweet_id,
                        "inference_results": completion,
                        "level": 2,
                    }
                ]
            )
            df_warnings = pd.concat([df_warnings, new_row], ignore_index=True)

    return df_warnings, (
        immediate_single,
        immediate_multiple,
        nonimmediate_single,
        nonimmediate_multiple,
        no_label,
        not_inferred,
    )


def detect_issue_type(completion):
    """
    Detect the type of issue in the completion.

    Args:
        completion (str): The completion string to check.

    Returns:
        str: The type of issue detected ("infinite_tagging", "restating_task", "meaningless_continuation", or "none").
    """
    if len(set(re.findall(r"@\w+", completion))) > 10:  # Detects infinite tagging
        return "infinite_tagging"
    elif re.search(
        r"Please classify the stance of the tweet below with respect to vaccination against human papill",
        completion,
    ):
        return "restating_task"
    elif len(set(re.findall(r"@\w+", completion))) > 10:
        return "meaningless_continuation"
    return "none"


def encode_stance_as_int(series):
    """
    Encode stance labels in a series to integers.

    Args:
        series (pd.Series): Series containing stance labels to encode.

    Returns:
        np.ndarray: Matrix of encoded stance labels.
    """

    def map_stance_to_int(stance):
        if Config.patterns[0][0] == stance:
            return 1  # in favor
        elif Config.patterns[1][0] == stance:
            return 2  # against
        elif Config.patterns[2][0] == stance:
            return 3  # neutral or unclear
        elif Config.patterns[3][0] == stance:
            return 0  # not inferred
        else:
            raise ValueError(f"Ill format: stance={stance}")

    mapped_results = series.str.lower().apply(map_stance_to_int)

    matrix = np.array(mapped_results).reshape((-1, Config.test_size))
    print("Encoded matrix shape:")
    print(matrix.shape)
    print("\n")
    assert matrix.shape == (len(series) / Config.test_size, Config.test_size)

    return matrix


def ensure_directory_exists_for_file(filepath):
    """
    Ensure the parent directory for a given file path exists.

    This function checks if the parent directory of the specified file path exists. If the directory does not exist, it creates the directory (including any necessary intermediate directories) to ensure that the file path is valid for file operations such as file creation or writing.

    Args:
        filepath (str): The complete file path for which the parent directory should be verified or created.
    """
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_prompt_to_messages(prompt):
    lines = prompt.strip().split("\n")

    messages = []
    problem_definition = ""
    is_problem_definition = True
    first_question = ""

    for line in lines:
        if line.startswith("Q:"):
            is_problem_definition = False
            if not first_question:
                first_question = line
            else:
                messages.append({"role": "user", "content": line})
        elif line.startswith("A:"):
            messages.append({"role": "assistant", "content": line})
        elif is_problem_definition:
            problem_definition += line + " "

    if first_question:
        first_message = problem_definition.strip() + " " + first_question
        messages.insert(0, {"role": "user", "content": first_message.strip()})

    return messages


def add_prompt_prefix(prompt):
    prompt = ast.literal_eval(prompt)
    prompt.append(
        {
            "role": "user",
            "content": "Please simply answer a label as short as possible without any complement, explanation, proof or inference. The stance of the above tweet is: ",
        }
    )

    return prompt


def check_and_recover_parallelized_order(df, df_processed):
    # - Check if the order of df has not changed after parallelized operations. Recover if not.
    id_column_original = df["id"].tolist()
    id_column_processed = df_processed["id"].tolist()

    sorting_is_maintained = id_column_original == id_column_processed

    if sorting_is_maintained:
        df = df_processed
    else:
        print("WARNING: unsorted completions after parallelized requests. Sorted now.")
        df_processed_sorted = df_processed.sort_values(by="id")
        df = df_processed_sorted

    return df


def assign_device(index, num_gpus=2):
    return f"cuda:{index % num_gpus}"
