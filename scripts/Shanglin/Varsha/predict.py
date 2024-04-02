# File: predict.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

from functools import partial
import os
import time
from typing import Tuple
import tiktoken
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import pandas as pd
from openai import OpenAI, RateLimitError
import backoff
from joblib import Parallel, delayed
import torch as th
from concurrent.futures import ThreadPoolExecutor
import json

from parameters import Config
from utils import *


class OpenAIClient:
    """Generate a static OpenAI() instance for each thread."""

    # --------------------
    # https://platform.openai.com/docs/guides/rate-limits/error-mitigation?context=tier-free
    # --------------------

    # use backoff to handle the rate limit error
    @staticmethod
    @backoff.on_exception(backoff.expo, RateLimitError)
    def completions_with_backoff(**kwargs):
        return OpenAI().chat.completions.create(**kwargs)


class FlanUL2Model:
    _model = None
    _tokenizer = None
    
    # lazy loading
    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
            cls._model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-ul2", device_map="auto", torch_dtype=th.bfloat16
            )

            cls._model.eval()
        return cls._model
    
    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
        return cls._tokenizer
            

class MistralModel:
    # tokenizer.pad_token = tokenizer.eos_token
    _model = None
    _tokenizer = None
    
    # lazy loading
    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            cls._model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", torch_dtype=th.bfloat16
            )

            cls._model.eval()
        return cls._model
    
    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        return cls._tokenizer


def loadTweets():
    """
    Load tweets from the CSV dataset.

    Returns:
        pd.DataFrame: A DataFrame containing tweets loaded from the CSV file.
                      It should have columns "id", "prompt", "tweet_id", "prompt_type", "true_label", "inference_results" and "notes".
                      It should have shape (1106, 7)
    """
    # Expected: The CSV file should only contain columns named "prompt", "tweet_id", "prompt_type" and "true_label".
    # Expected: The CSV file should contain 1106 rows for 1106 prompts.
    if os.path.isfile(Config.checkpoint_results_file) and request_user_confirmation(
        f"Found checkpoint file {Config.checkpoint_results_file}. {Color.YELLOW}Append?{Color.END} [Y/n]: ",
        "Appending...",
        "Ignored checkpoint file."
    ):
        df = pd.read_csv(
            Config.checkpoint_results_file,
            dtype={
                "inference_results": "object",
                "notes": "object"
            }
        )
        
        index = df['inference_results'].first_valid_index()
        
        if index is not None:
            # the index of the first empty value after the first valid value
            for i in range(index, len(df['inference_results'])):
                if pd.isna(df['inference_results'][i]):
                    cp_length = i
                    break
            else:
                cp_length = len(df)
        else:
            # empty column "inference_results"
            cp_length = 0

        print(f"Checkpoint length: {cp_length}")
        
        return df, cp_length
    
    df = pd.read_csv(Config.dataset_file)

    df["inference_results"] = pd.Series(dtype="string")
    df["notes"] = pd.Series(dtype="string")

    # ------------------------------------------------------------------
    # Generate an "id" column to to record the original order of prompts.
    # This is crucial for parallelization,
    # as parallel processing may alter the original sequence of prompts.
    # ------------------------------------------------------------------

    # reset_index() extracts index as a separate column "df_reset" whose index starts from 0
    df_reset = df.reset_index()

    # make the new column's index start from 1 and insert it to the leftmost of the existing df
    df.insert(0, "id", df_reset.index + 1)

    return df, 0


def generate_test_csv():
    """Extract first 14 rows from complete dataset file and store them in a test dataset file. Call this function ONLY when Config.test_dataset_filename does not exist."""
    df = pd.read_csv(Config.complete_dataset_file)
    subset_df = df.head(Config.test_size)

    ensure_directory_exists_for_file(Config.test_dataset_file)

    subset_df.to_csv(Config.test_dataset_file, index=False)


def estimate_for_gpt(df_prompts):
    """Estimate accumulated token number and cost for all prompts. Adapted from cost_est_auto_0216.py.

    Args:
        df_prompts (df): a df with a single column, containing all prompts.
    """

    def calculate_cost(encoding, prompt):
        """Calculate the number of tokens and cost for a given prompt string.

        Args:
            encoding (tiktoken.Encoding): The encoding object for tokenization.
            prompt (str): The prompt for which to calculate the cost.

        Returns:
            Tuple[int, float]: A tuple containing the number of tokens and the cost for the prompt.
        """
        tokens = encoding.encode(prompt)
        num_tokens = len(tokens)
        cost = num_tokens * Config.cost_per_token

        return num_tokens, cost

    encoding = tiktoken.encoding_for_model(Config.model)
    # Create a partial function to fix the 'encoding' parameter in calculate_cost
    calculate_cost_partial = partial(calculate_cost, encoding)

    # parallelization, actually not necessary
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(calculate_cost_partial, df_prompts))

    total_tokens, total_cost = map(sum, zip(*results))

    return (total_tokens, total_cost)


def run_gpt_model_for_one_batch(df_batch):
    """Label the stances of the prompts with GPT models. This involves interacting with the model with OpenAI's API. This is the helper function that should be called by `run_gpt_model`. It enables parallelization. Adapted from Sean's.

    Args:
        df_chunk (pd.DataFrame): the df that contains the prompts to be fed into models. df_chunk has the shape of (1106 / num_cores, 7). All operations directely update df_chunk.

    Returns:
        df_chunk (pd.DataFrame): The same DataFrame as the input argument.
    """
    def send_request(prompt):
        """Send a request to OpenAI's chat API to generate a response based on the given prompt string.

        Args:
            prompt (str): The input prompt for generating the response.

        Returns:
            dict: A dictionary containing the response generated by the chat API.
        """
        # --------------------
        # https://platform.openai.com/docs/guides/text-generation/chat-completions-api
        # --------------------
        chat = OpenAIClient.completions_with_backoff(
            model=Config.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=Config.max_output_tokens,
        )
        return chat

    # Get the beginning of the default DataFrame index (not a user-defined 'id' column).
    # We need to record this for df updates later on.
    starting_index = df_batch.index[0]

    # index stands for the iteration index and ascend by 1 each time
    # id stands for 'id' column and might have been shuffled here
    for index, (id, prompt) in enumerate(zip(df_batch["id"], df_batch["prompt"])):
        try:
            chat = send_request(prompt)
        except Exception as e:
            message_warning = f"Tweet ID: {id}. Error: {e} "
            print(message_warning)

            df_batch.loc[starting_index + index, "notes"] = message_warning

            time.sleep(30)  # retry after 30s, unsure if we can reduce to 10s
            chat = send_request(prompt)
        # --------------------
        # Parse the output and collect the results
        # --------------------
        output_label = chat.choices[0].message.content
        if chat.choices[0].finish_reason != "stop":
            message_warning = (
                f"Tweet ID: {id}. Warning: the prediction is not finished."
            )
            print(message_warning)

            df_batch.loc[starting_index + index, "notes"] = message_warning
            
        print(f"tweet {index + 1}/{len(df_batch)}: {output_label}")
        
        df_batch.loc[starting_index + index, "inference_results"] = output_label

    return df_batch


def run_model_in_parallel(df):
    """Send prompts to GPT API using parallelization and retrieve inference results.

    Args:
        df_prompts (pd.DataFrame): A DataFrame containing 1106 rows and 7 columns (id, prompt, tweet_id, prompt_type, true_label, inference_results, notes). The "inference_results" and "notes" columns will be updated.

    Raises:
        NotImplementedError: Raised when non-parallelization GPT inference logics need optimization.
        ValueError: Raised if Config.max_concurrent_api_calls is not a positive integer.

    Returns:
        pd.DataFrame: The original DataFrame (df_prompts) with updated "inference results" and "notes" columns after sending prompts to the GPT API.
    """
    num_cores = min(Config.max_concurrent_api_calls, len(df))

    # split the input df into chunks, obtaining a list containing df chunks
    # each df_chunk has the shape of around (1106 / Config.max_concurrent_api_calls, 4)
    list_df_chunks = np.array_split(df, num_cores)

    # - use Parallel and delayed to execute the function in parallel
    # update list_df_chunks as a list containing all "df_chunk"s returned by "run_gpt_model_on_one_core"s
    # the order of list_df_chunks does not change.

    # --------------------
    # WARNING: Cost Incurred Below
    # --------------------

    try:
        list_df_chunks = Parallel(n_jobs=num_cores)(
            delayed(run_model_on_one_core)(df_chunk) for df_chunk in list_df_chunks
        )            
        return pd.concat(list_df_chunks), None
        
    except Exception as e:
        return pd.concat(list_df_chunks), str(e)


def run_model_on_one_core(df_core):
    # divide the tweets into batches
    # note that the last batch might be smaller than batch_size
    for batch_count, start_index in enumerate(
        range(0, len(df_core), Config.batch_size), start=1
    ):
        end_index = start_index + Config.batch_size
        df_batch = df_core.iloc[start_index:end_index]

        # filter out rows with excluded prompt types
        excluded_rows_indices = df_batch["prompt_type"].isin(
            Config.excluded_prompts[Config.model]
        )
        df_batch.loc[excluded_rows_indices, "inference_results"] = "not inferenced"
        df_batch = df_batch[~excluded_rows_indices]

        print(f"Processing batch {batch_count}...")

        if df_batch.shape[0] == 0:
            continue

        # update df_batch's "inference_results" and "notes" columns
        if Config.model_abbr == "FLAN-UL2":
            df_batch = run_flan_model_for_one_batch(df_batch)
        elif Config.model_abbr == "MISTRAL":
            df_batch = run_mistral_model_for_one_batch(df_batch)
        elif Config.model_abbr == "GPT4":
            df_batch = run_gpt_model_for_one_batch(df_batch)
        else:
            raise NotImplementedError(f"Model not implemented yet: {Config.model_abbr}")
            
        # update df
        
        df_core.loc[df_batch.index, "inference_results"] = df_batch["inference_results"]
        df_core.loc[df_batch.index, "notes"] = df_batch["notes"]

    return df_core


def run_flan_model_for_one_batch(df_batch):
    """
    Generates predictions for the given input texts using the specified FLAN model.

    Args:
        df_prompts (pd.DataFrame): A DataFrame containing <= BATCH_SIZE rows and 7 columns (id, prompt, tweet_id, prompt_type, true_label, inference_results, notes). The "inference_results" and "notes" columns will be updated.

    Returns:
        pd.DataFrame: The input DataFrame with "inference_results" and "notes" columns updated.
    """
    with th.no_grad():
        try:
            input_ids = FlanUL2Model.get_tokenizer()(
                list(df_batch["prompt"]), return_tensors="pt"
            ).input_ids.to(Config.device)

            outputs = FlanUL2Model.get_model().generate(
                input_ids, max_new_tokens=Config.max_output_tokens
            )
            output_labels = FlanUL2Model.get_tokenizer().batch_decode(
                outputs, skip_special_tokens=True
            )

            df_batch.loc[:, "inference_results"] = output_labels
            # print(output_labels)

        except Exception as e:
            message_warning = f"Batch: {(df_batch['id'][0] - 1) // Config.batch_size + 1}. Error: {e} "
            print(message_warning)
            df_batch["notes"] += message_warning

    return df_batch


def run_mistral_model_for_one_batch(df_batch):
    """
    Generates predictions for the given input texts using the specified FLAN model.

    Args:
        df_prompts (pd.DataFrame): A DataFrame containing <= BATCH_SIZE rows and 7 columns (id, prompt, tweet_id, prompt_type, true_label, inference_results, notes). The "inference_results" and "notes" columns will be updated.

    Returns:
        pd.DataFrame: The input DataFrame with "inference_results" and "notes" columns updated.
    """
    with th.no_grad():
        # try:
        # input_ids = MistralModel.tokenizer(
        #     list(df_batch["prompt"]), padding=True, return_tensors="pt"
        # ).input_ids.to(Config.device)

        # outputs = MistralModel.model.generate(
        #     input_ids,
        #     max_new_tokens=Config.max_output_tokens,
        #     pad_token_id=MistralModel.tokenizer.eos_token_id,
        #     do_sample=True
        # )
        # output_labels = MistralModel.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(output_labels)
        # input()

        # https://huggingface.co/docs/transformers/main/en/model_doc/mistral#transformers.MistralForCausalLM

        # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

        # https://discuss.huggingface.co/t/generate-returns-full-prompt-plus-answer/70453

        # File "/home/syang662/miniconda3/envs/llm/lib/python3.11/site-packages/transformers/generation/utils.py", line 1376, in generate

        # https://huggingface.co/docs/transformers/main/en/chat_templating
        
        try:
            encoded_embedded_batch = [
                MistralModel.get_tokenizer().apply_chat_template([
                        {"role": "user", "content": f"{prompt} Simply answer a label without any explanation. The stance of the above tweet is: "}
                    ],
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(Config.device)
                for prompt in df_batch["prompt"]
            ]

            stacked_batch = th.stack(encoded_embedded_batch, dim=0).squeeze(0)

            generated_ids = MistralModel.get_model().generate(
                stacked_batch,
                max_new_tokens=1000,
                do_sample=True,
                pad_token_id=MistralModel.get_tokenizer().eos_token_id,
            )
            decoded = MistralModel.get_tokenizer().batch_decode(generated_ids)
            
            extracted_outputs = [
                extract_answer_from_mistral_raw_completions(completion)
                for completion in decoded
            ]
            print(extracted_outputs)

            df_batch.loc[:, "inference_results"] = extracted_outputs

        except Exception as e:
            message_warning = f"Batch: {(df_batch['id'].iloc[0] - 1) // Config.batch_size + 1}. Error: {e} "
            print(message_warning)
            df_batch["notes"] += message_warning

    return df_batch


def extract_answer_from_mistral_raw_completions(model_output):
    # 定义开始和结束标记
    unk_token = "<unk>"
    start_token = "[/INST]"
    end_token = "</s>"

    # 尝试找到最后一个<unk>标记的位置
    last_unk_index = model_output.rfind(unk_token)

    # 如果找到了<unk>标记，调整开始位置到<unk>之后
    if last_unk_index != -1:
        start_index = last_unk_index + len(unk_token)
    else:
        # 如果没有找到<unk>标记，寻找最后一处A:作为开始
        last_start_index = model_output.rfind(start_token)
        if last_start_index != -1:
            start_index = last_start_index + len(start_token)
        else:
            # 如果既没有<unk>也没有A:，可能是异常情况
            return "Unable to find the start of the answer."

    # 查找结束标记的位置
    end_index = model_output.find(end_token, start_index)

    # 如果找到了结束标记
    if end_index != -1:
        return model_output[start_index:end_index].strip()
    else:
        # 如果没有找到结束标记，返回从开始标记之后的所有文本
        return model_output[start_index:].strip()


def predict() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Predict using the specified model and process tweets.

    Raises:
        NotImplementedError: Raised if the inference logic for the 'flan' model needs optimization.

    Returns:
        pd.DataFrame: The processed DataFrame containing columns ('id', 'prompt', 'tweet_id', 'prompt_type', 'true_label', 'inference_results', 'notes'). Shape should be (1106, 7).
        pd.DataFrame: The DataFrame containing rows with non-empty 'notes' column, if any. Columns are the same as previous df. Row number undetermined.
    """
    ensure_dataset_file_exists()

    # columns = [id, prompt, tweet_id, prompt_type, true_label, inference_results, notes], #rows = 1106 = 79 * 14
    df_checkpoint, checkpoint_length = loadTweets()

    if checkpoint_length:
        print(f"Start processing Tweets in {Config.dataset_file} with {Config.model} resuming from {Config.checkpoint_results_file}.\n")
        df = df_checkpoint.iloc[checkpoint_length:]
    else:
        print(f"Start processing Tweets in {Config.dataset_file} with {Config.model}.\n")
        df = df_checkpoint

    memory_usage = df.memory_usage(deep=True).sum()
    print(f"Approximate memory usage for df: {memory_usage / 1024 ** 2} MB")

    if Config.model_abbr in ["FLAN_UL2", "MISTRAL"]:
        print(f"Device: {Config.device}")

    elif Config.model_abbr in ["GPT4"]:
        # check price and ask for confirmation
        total_tokens, total_cost = estimate_for_gpt(df["prompt"])
        print(
            f"Estimated total tokens = {Color.RED}{total_tokens}{Color.END} and total cost = ${Color.RED}{total_cost:.2f}{Color.END}."
        )

        if not request_user_confirmation(
            f"{Color.YELLOW}Continue?{Color.END} [Y/n]: ",
            "Continuing...",
            "Terminated.",
        ):
            exit(0)

    else:
        raise NotImplementedError(f"Unexpected model: {Config.model}")

    df_processed, error = run_model_in_parallel(df)
    
    df = check_and_recover_parallelized_order(df, df_processed)

    df_warnings = check_completion_format(df)
    df_warnings.to_csv(Config.format_warnings_file, index=False)
    
    if checkpoint_length:
        df = pd.concat([df_checkpoint.iloc[:checkpoint_length], df], ignore_index=True)
    
    if error:
        print(f"\nTerminated processing {Config.dataset_file} with {Config.model}. Error:\n{error}")
        df_with_not_empty_notes = save_results_and_extract_warnings(df, True)
    else:
        print(f"\nFinished processing {Config.dataset_file} with {Config.model}.")
        df_with_not_empty_notes = save_results_and_extract_warnings(df, False)

    return df, df_with_not_empty_notes


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


def save_results_and_extract_warnings(df, checkpoint):
    results_file = Config.checkpoint_results_file if checkpoint else Config.results_file
    warnings_file = Config.checkpoint_warnings_file if checkpoint else Config.warnings_file
    
    ensure_directory_exists_for_file(results_file)

    df.to_csv(results_file, index=False)
    print(f"\nResults stored in {results_file}.\n")

    # pick df rows with nonempty notes (if any) and store them in a separate csv
    df_with_non_empty_notes = df[df["notes"].notna()]

    if not df_with_non_empty_notes.empty:
        ensure_directory_exists_for_file(Config.warnings_file)
        df_with_non_empty_notes.to_csv(Config.warnings_file, index=False)
        print(
            f"WARNING: Rows with not empty 'notes' column saved to {warnings_file}"
        )
    else:
        print("Good news! No rows with not empty 'notes' column found.")

    return df_with_non_empty_notes


def ensure_dataset_file_exists():
    # e.g., dataset/Varsha/all_prompts_0226.csv
    if not os.path.isfile(Config.dataset_file):
        if Config.is_test_run:
            generate_test_csv()
        else:
            raise FileNotFoundError(f"{Config.dataset_file} does not exist!")
            
            
def save_to_json(df_notes, time_):
    config_instance = Config()
    
    report_dict = {
        attr: getattr(config_instance, attr) for attr in dir(config_instance) if not attr.startswith('__') and not callable(getattr(config_instance, attr))
    }
    
    report_dict["device"] = str(report_dict["device"])
    
    notes_list_dict = df_notes.to_dict('records')
    
    report_dict['df_with_not_empty_notes'] = notes_list_dict
    report_dict['time'] = time_
    del report_dict['openai_key']
    
    with open(Config.execution_report_file, 'w') as json_file:
        json.dump(report_dict, json_file, indent=4)
        
    print("\n-------------------error messages-------------------")
    print(df_notes, end="\n\n")

    print("-----------------time-----------------")
    print(f"time: {time_:.2f}s")


if __name__ == "__main__":
    t1 = time.time()

    # e.g., results/Varsha/GPT4/gpt-4-0125-preview.csv
    if os.path.exists(Config.results_file):
        print(f"{Config.results_file} already exists.\n")

        if not request_user_confirmation(
            f"{Color.YELLOW}Overwrite?{Color.END} [Y/n]: ",
            "Overwriting...",
            f"{Config.results_file} is skipped.",
        ):
            exit(0)

    df, df_with_not_empty_notes = predict()

    t2 = time.time()
    
    save_to_json(df_with_not_empty_notes.loc[:, ["id", "inference_results", "notes"]], t2 - t1)
