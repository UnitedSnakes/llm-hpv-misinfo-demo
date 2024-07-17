# File: predict.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

from functools import partial
import multiprocessing
import os
import time
from typing import Tuple
import tiktoken
from transformers import (
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import numpy as np
import pandas as pd
import pandas as pd
from openai import OpenAI, RateLimitError
import backoff
import torch as th
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import argparse
import signal

from config import Config
from utils import *


class SignalHandler:
    def __init__(self):
        self._signal_received = multiprocessing.Manager().Event()  # Use Event from Manager for boolean flag

    def signal_handler(self, signum, frame):
        print(f"Received signal: {signum}")
        self._signal_received.set()  # Set the event to indicate signal received

    def signal_received(self):
        if self._signal_received.is_set():
            print("Received stop signal, terminating process...")
            return True
        return False


class OpenAIClient:
    """Generate a static OpenAI() instance for each thread."""

    if Config.model.startswith("gpt"):
        encoding = tiktoken.get_encoding(Config.model)

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
            assert "flan" in Config.model
            cls._tokenizer = AutoTokenizer.from_pretrained(Config.model)
            cls._model = T5ForConditionalGeneration.from_pretrained(
                Config.model, device_map="auto", torch_dtype=th.bfloat16
            )

            print(Config.model)

            cls._model.eval()
        return cls._model

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            assert "flan" in Config.model
            cls._tokenizer = AutoTokenizer.from_pretrained(Config.model)
        return cls._tokenizer


class MistralModel:
    _model = None
    _tokenizer = None

    # lazy loading
    @classmethod
    def get_model(cls, device):
        if cls._model is None:
            assert "mistral" in Config.model
            cls._tokenizer = cls.get_tokenizer()
                
            quantization_config = None
            if Config.load_in_4bit or Config.load_in_8bit:
                assert "8x7B" in Config.model
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=Config.load_in_4bit,
                    load_in_8bit=Config.load_in_8bit,
                )

            cls._model = AutoModelForCausalLM.from_pretrained(
                Config.model,
                # device_map="auto",
                device_map={"":device},
                torch_dtype="auto" if quantization_config else th.float16,
                quantization_config=quantization_config,
                token=os.getenv("HF_ACCESS_TOKEN"),
            )

            cls._model.eval()
        return cls._model

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            assert "mistral" in Config.model
            cls._tokenizer = AutoTokenizer.from_pretrained(
                Config.model, token=os.getenv("HF_ACCESS_TOKEN"), padding_side="left"
            )
            # cls._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            # cls._tokenizer.padding_side = 'left'
            # cls._tokenizer.pad_token_id = cls._tokenizer.eos_token_id
        return cls._tokenizer


class LlamaModel:
    _model = None
    _tokenizer = None
    _terminator = None

    # lazy loading
    @classmethod
    def get_model(cls, device):
        if cls._model is None:
            assert "llama" in Config.model
            cls._tokenizer = cls.get_tokenizer()
            
            quantization_config = None
            if Config.load_in_4bit or Config.load_in_8bit:
                assert "70B" in Config.model
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=Config.load_in_4bit,
                    load_in_8bit=Config.load_in_8bit,
                )
            cls._model = AutoModelForCausalLM.from_pretrained(
                Config.model,
                device_map={"":device},
                torch_dtype="auto" if quantization_config else th.float16,
                quantization_config=quantization_config,
                token=os.getenv("HF_ACCESS_TOKEN"),
            )

            cls._model.eval()
        return cls._model

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            assert "llama" in Config.model
            cls._tokenizer = AutoTokenizer.from_pretrained(
                Config.model, token=os.getenv("HF_ACCESS_TOKEN"), padding_side = "left"
            )
            cls._tokenizer.pad_token_id = cls.get_terminator()[0]

        return cls._tokenizer

    @classmethod
    def get_terminator(cls):
        if cls._terminator is None:
            assert "llama" in Config.model
            cls._terminator = [
                cls.get_tokenizer().eos_token_id,
                cls.get_tokenizer().convert_tokens_to_ids("<|eot_id|>"),
            ]
        return cls._terminator


class PromptProcessor:
    longest_prompt_id = None
    longest_token_length = 0

    @classmethod
    def update_longest_prompt(cls, id, tokenized_prompt_length):
        if tokenized_prompt_length > cls.longest_token_length:
            cls.longest_token_length = tokenized_prompt_length
            cls.longest_prompt_id = id

    @classmethod
    def get_longest_prompt_id(cls):
        return cls.longest_prompt_id

    @classmethod
    def get_longest_prompt_length(cls):
        return cls.longest_token_length


def loadTweets():
    """
    Load tweets from the CSV dataset.

    Returns:
        pd.DataFrame: A DataFrame containing tweets loaded from the CSV file.
                      It should have columns "id", "prompt", "tweet_id", "prompt_type", "true_label", "inference_results" and "notes".
                      It should have shape (1106, 7)
    """
    ensure_directory_exists_for_file(Config.dataset_file)
    df = pd.read_csv(Config.dataset_file)

    if Config.reinfer:
        columns_to_keep = ["prompt", "tweet_id", "prompt_type", "true_label"]
        df = df[columns_to_keep]

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

    return df


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
    if df_prompts.empty:
        print("The DataFrame is empty. Returning (0, 0.0).")
        return (0, 0.0)

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
        token_ids = OpenAI.encoding.encode(prompt)
        PromptProcessor.update_longest_prompt(id, len(token_ids))
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
    if df.empty:
        print(
            "The DataFrame is empty. Returning an empty DataFrame and None error message."
        )
        return df, None

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

    # try:    
        # Initialize SignalHandler
    signal_handler = SignalHandler()
    signal.signal(signal.SIGUSR1, signal_handler.signal_handler)

    # Use ProcessPoolExecutor to execute the function in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(run_model_on_one_core, df_chunk, signal_handler, assign_device(i)) for i, df_chunk in enumerate(list_df_chunks)]
        list_df_chunks = [future.result() for future in futures]
    
    if signal_handler.signal_received():
        return pd.concat(list_df_chunks), "Received stop signal, terminating process..."
    return pd.concat(list_df_chunks), None

    # except Exception as e:
    #     return pd.concat(list_df_chunks), str(e)


def run_model_on_one_core(df_core, signal_handler, device):    
    for batch_count, start_index in enumerate(
        range(0, len(df_core), Config.batch_size), start=1
    ):
        if signal_handler.signal_received():
            break

        end_index = start_index + Config.batch_size
        df_batch = df_core.iloc[start_index:end_index]

        print(f"Processing batch {batch_count}...")

        if df_batch.shape[0] == 0:
            continue

        # update df_batch's "inference_results" and "notes" columns
        if Config.model_abbr.startswith("FLAN_UL2"):
            df_batch = run_flan_model_for_one_batch(df_batch, device)
        elif Config.model_abbr.startswith("MISTRAL"):
            df_batch = run_mistral_model_for_one_batch(df_batch, device)
        elif Config.model_abbr.startswith("LLAMA3"):
            df_batch = run_llama_model_for_one_batch(df_batch, device)
        elif Config.model_abbr.startswith("GPT4"):
            df_batch = run_gpt_model_for_one_batch(df_batch)
        else:
            raise NotImplementedError(f"Model not implemented yet: {Config.model_abbr}")

        # update df
        df_core.loc[df_batch.index, "inference_results"] = df_batch["inference_results"]
        df_core.loc[df_batch.index, "notes"] = df_batch["notes"]

    return df_core


@th.no_grad()
def run_flan_model_for_one_batch(df_batch):
    """
    Generates predictions for the given input texts using the specified FLAN model.

    Args:
        df_prompts (pd.DataFrame): A DataFrame containing <= BATCH_SIZE rows and 7 columns (id, prompt, tweet_id, prompt_type, true_label, inference_results, notes). The "inference_results" and "notes" columns will be updated.

    Returns:
        pd.DataFrame: The input DataFrame with "inference_results" and "notes" columns updated.
    """
    batch_index = (df_batch["id"].iloc[0] - 1) // Config.batch_size + 1
    try:
        input_ids = FlanUL2Model.get_tokenizer()(
            list(df_batch["prompt"]), return_tensors="pt"
        ).input_ids.to(Config.device)

        PromptProcessor.update_longest_prompt(
            df_batch["id"].iloc[0], input_ids.shape[1]
        )

        # Check if any of the tokenized inputs are too long
        if input_ids.shape[1] > FlanUL2Model.get_tokenizer().model_max_length:
            warning_message = f"Batch: {batch_index}. Input is too long. Received {input_ids.size(1)} tokens. Model max length is {FlanUL2Model.get_tokenizer().model_max_length} tokens. Marked as 'not inferred'."

            print(warning_message)

            df_batch.loc[:, "inference_results"] = (
                f"not inferred: exceeding model input tokens limit ({input_ids.size(1)} > {FlanUL2Model.get_tokenizer().model_max_length})"
            )

            return df_batch

        outputs = FlanUL2Model.get_model().generate(
            input_ids,
            max_new_tokens=Config.max_output_tokens,
            do_sample=True,
            temperature=Config.temperature,
        )
        output_labels = FlanUL2Model.get_tokenizer().batch_decode(
            outputs, skip_special_tokens=True
        )

        df_batch.loc[:, "inference_results"] = output_labels
        print(output_labels)

        th.cuda.empty_cache()

    except Exception as e:
        message_warning = f"Batch: {batch_index}. Error: {e} "
        print(message_warning)
        df_batch.loc[:, "notes"] += message_warning

    return df_batch


def add_inst_tokens_to_qa_pairs(prompt: str) -> str:
    """
    Add [INST] and [/INST] tokens around each Q: and A: pair in the prompt.

    Args:
        prompt (str): The original prompt containing multiple Q: and A: pairs.

    Returns:
        str: The prompt with [INST] and [/INST] tokens added around each Q: and A: pair.
    """
    lines = prompt.split("\n")
    new_lines = []
    in_qa_pair = False

    for line in lines:
        if line.startswith("Q:"):
            if in_qa_pair:
                new_lines.append("[/INST]")
            new_lines.append("[INST] " + line)
            in_qa_pair = True
        elif line.startswith("A:"):
            new_lines.append(line + " [/INST]")
            in_qa_pair = False
        else:
            new_lines.append(line)

    if in_qa_pair:
        new_lines.append("[/INST]")

    return "\n".join(new_lines)


@th.no_grad()
def run_mistral_model_for_one_batch(df_batch, device):
    """
    Generates predictions for the given input texts using the specified FLAN model.

    Args:
        df_prompts (pd.DataFrame): A DataFrame containing <= BATCH_SIZE rows and 7 columns (id, prompt, tweet_id, prompt_type, true_label, inference_results, notes). The "inference_results" and "notes" columns will be updated.

    Returns:
        pd.DataFrame: The input DataFrame with "inference_results" and "notes" columns updated.
    """
    # https://huggingface.co/docs/transformers/main/en/model_doc/mistral#transformers.MistralForCausalLM

    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

    # https://discuss.huggingface.co/t/generate-returns-full-prompt-plus-answer/70453

    # https://huggingface.co/docs/transformers/main/en/chat_templating

    batch_index = (df_batch["id"].iloc[0] - 1) // Config.batch_size + 1

    # try:
    if Config.add_prompt_prefix:
        df_batch.loc[:, "prompt"] = df_batch["prompt"].apply(add_prompt_prefix)
    else:
        df_batch.loc[:, "prompt"] = df_batch["prompt"].apply(convert_prompt_to_messages)

    input_ids = [
        MistralModel.get_tokenizer().apply_chat_template(prompt, return_tensors="pt")
        for prompt in df_batch["prompt"]
    ]

    input_ids = th.stack(input_ids).to(Config.device).squeeze(0)

    PromptProcessor.update_longest_prompt(df_batch["id"].iloc[0], input_ids.shape[1])

    # Check if any of the tokenized input_ids are too long
    if input_ids.shape[1] > 8192:
        warning_message = f"Batch: {batch_index}. Input is too long. Received {input_ids.size(1)} tokens. Model max length is {LlamaModel.get_tokenizer().model_max_length} tokens. Marked as 'not inferred'."

        print(warning_message)

        df_batch.loc[:, "inference_results"] = (
            f"not inferred: exceeding model input tokens limit ({input_ids.size(1)} > {LlamaModel.get_tokenizer().model_max_length})"
        )

        return df_batch

    outputs = MistralModel.get_model(device).generate(
        input_ids,
        pad_token_id=MistralModel.get_tokenizer().eos_token_id,
        max_new_tokens=Config.max_output_tokens,
        temperature=Config.temperature,
        do_sample=True,
    )

    responses = MistralModel.get_tokenizer().batch_decode(
        outputs, skip_special_tokens=True
    )
    
    # Remove the prompt from the start of the output
    # anchor = "[/INST]"
    # anchor = "[/INST]A:"
    # clean_responses = [
    #     response[response.rfind(anchor) + len(anchor) :].strip() for response in responses
    # ]

    responses = [output[input_ids.shape[-1]:] for output in outputs]

    clean_responses = [
        MistralModel.get_tokenizer().decode(response, skip_special_tokens=True).strip()
        for response in responses
    ]

    print(clean_responses)

    df_batch.loc[:, "inference_results"] = clean_responses

    # mistral might have memory leak or what...GPU will run out of RAM if ignore this line :(
    th.cuda.empty_cache()

    # except Exception as e:
        # message_warning = f"Batch: {batch_index}. Error: {e} "
        # print(message_warning)
        # df_batch.loc[:, "notes"] += message_warning

    return df_batch


@th.no_grad()
def run_llama_model_for_one_batch(df_batch, device):
    """
    Generates predictions for the given input texts using the specified LLAMA model.

    Args:
        df_prompts (pd.DataFrame): A DataFrame containing <= BATCH_SIZE rows and 7 columns (id, prompt, tweet_id, prompt_type, true_label, inference_results, notes). The "inference_results" and "notes" columns will be updated.

    Returns:
        pd.DataFrame: The input DataFrame with "inference_results" and "notes" columns updated.
    """
    batch_index = (df_batch["id"].iloc[0] - 1) // Config.batch_size + 1

    # try:
    # prefixed_prompts = [
    #     f"{prompt} Please simply answer a label as short as possible without any complement, explanation, proof or inference. The stance of the above tweet is: "
    #     for prompt in df_batch["prompt"]
    # ]
    if Config.add_prompt_prefix:
        df_batch.loc[:, "prompt"] = df_batch["prompt"].apply(add_prompt_prefix)
    else:
        df_batch.loc[:, "prompt"] = df_batch["prompt"].apply(convert_prompt_to_messages)

    input_ids = [
        LlamaModel.get_tokenizer().apply_chat_template(prompt, return_tensors="pt").to(device)
        for prompt in df_batch["prompt"]
    ]

    input_ids = th.stack(input_ids).to(device).squeeze(0)

    PromptProcessor.update_longest_prompt(df_batch["id"].iloc[0], input_ids.shape[1])

    # Check if any of the tokenized inputs are too long
    if input_ids.shape[1] > 4096:
        warning_message = f"Batch: {batch_index}. Input is too long. Received {input_ids.size(1)} tokens. Model max length is {LlamaModel.get_tokenizer().model_max_length} tokens. Marked as 'not inferred'."

        print(warning_message)

        df_batch.loc[:, "inference_results"] = (
            f"not inferred: exceeding model input tokens limit ({input_ids.size(1)} > {LlamaModel.get_tokenizer().model_max_length})"
        )

        return df_batch

    outputs = LlamaModel.get_model(device).generate(
        input_ids,
        max_new_tokens=Config.max_output_tokens,
        eos_token_id=LlamaModel.get_terminator(),
        pad_token_id=LlamaModel.get_tokenizer().eos_token_id,
        do_sample=True,
        temperature=Config.temperature,
    )

    responses = [output[input_ids.shape[-1]:] for output in outputs]

    clean_responses = [
        LlamaModel.get_tokenizer().decode(response, skip_special_tokens=True).strip()
        for response in responses
    ]
    
    print(clean_responses)

    df_batch.loc[:, "inference_results"] = clean_responses

    th.cuda.empty_cache()

    # except Exception as e:
    #     message_warning = f"Batch: {batch_index}. Error: {e} "
    #     print(message_warning)
    #     df_batch.loc[:, "notes"] += message_warning

    return df_batch


def extract_answer_from_mistral_raw_completions(model_output):
    # Define the unknown and start/end tokens
    unk_token = "<unk>"
    start_token = "[/INST]"
    end_token = "</s>"

    # Attempt to find the position of the last <unk> token
    last_unk_index = model_output.rfind(unk_token)

    # If an <unk> token was found, adjust the starting position to just after <unk>
    if last_unk_index != -1:
        start_index = last_unk_index + len(unk_token)
    else:
        # If no <unk> token was found, look for the last occurrence of A: as the start
        last_start_index = model_output.rfind(start_token)
        if last_start_index != -1:
            start_index = last_start_index + len(start_token)
        else:
            # If neither <unk> nor A: was found, it may be an exceptional case
            return "Unable to find the start of the answer."

    # Locate the position of the end token
    end_index = model_output.find(end_token, start_index)

    # If an end token was found
    if end_index != -1:
        return model_output[start_index:end_index].strip()
    else:
        # If no end token was found, return all text from the start token onward
        return model_output[start_index:].strip()


def predict(df, df_checkpoint, checkpoint_length) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Predict using the specified model and process tweets.

    Raises:
        NotImplementedError: Raised if the inference logic for the 'flan' model needs optimization.

    Returns:
        pd.DataFrame: The processed DataFrame containing columns ('id', 'prompt', 'tweet_id', 'prompt_type', 'true_label', 'inference_results', 'notes'). Shape should be (1106, 7).
        pd.DataFrame: The DataFrame containing rows with non-empty 'notes' column, if any. Columns are the same as previous df. Row number undetermined.
    """
    memory_usage = df.memory_usage(deep=True).sum()
    print(f"Approximate memory usage for df: {memory_usage / 1024 ** 2} MB")

    if Config.model_abbr.startswith((
        "FLAN_UL2",
        "MISTRAL",
        "LLAMA3",
    )):
        print(f"Device: {Config.device}")

    elif Config.model_abbr.startswith("GPT4"):
        # check price and ask for confirmation
        total_tokens, total_cost = estimate_for_gpt(df["prompt"])

        Config.total_cost = total_cost

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

    df_warnings, _ = check_completion_format(df)
    ensure_directory_exists_for_file(Config.format_warnings_file)
    df_warnings.to_csv(Config.format_warnings_file, index=False)
    
    print(df_warnings)

    if checkpoint_length:
        df = pd.concat([df_checkpoint.iloc[:checkpoint_length], df], ignore_index=True)

    if error:
        print(
            f"\nTerminated processing {Config.dataset_file} with {Config.model}. Error:\n{error}"
        )
        df_with_not_empty_notes = save_results_and_extract_warnings(df, True)
    else:
        print(f"\nFinished processing {Config.dataset_file} with {Config.model}.")
        df_with_not_empty_notes = save_results_and_extract_warnings(df, False)

    return df, df_with_not_empty_notes


def save_results_and_extract_warnings(df, checkpoint):
    results_file = Config.checkpoint_results_file if checkpoint else Config.results_file
    warnings_file = (
        Config.checkpoint_warnings_file if checkpoint else Config.warnings_file
    )

    ensure_directory_exists_for_file(results_file)

    df.to_csv(results_file, index=False)
    print(f"\nResults stored in {results_file}.\n")

    # pick df rows with nonempty notes (if any) and store them in a separate csv
    df_with_non_empty_notes = df[df["notes"].notna()]

    if not df_with_non_empty_notes.empty:
        ensure_directory_exists_for_file(Config.warnings_file)
        df_with_non_empty_notes.to_csv(Config.warnings_file, index=False)
        print(f"WARNING: Rows with not empty 'notes' column saved to {warnings_file}")
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
        attr: getattr(config_instance, attr)
        for attr in dir(config_instance)
        if not attr.startswith("__") and not callable(getattr(config_instance, attr))
    }

    report_dict["device"] = str(report_dict["device"])

    notes_list_dict = df_notes.to_dict("records")

    report_dict["df_with_not_empty_notes"] = notes_list_dict
    report_dict["time"] = time_

    longest_prompt_id = PromptProcessor.get_longest_prompt_id()
    if longest_prompt_id:
        longest_prompt_id = int(longest_prompt_id)
    report_dict["longest_prompt_id"] = PromptProcessor.get_longest_prompt_id()

    longest_prompts = PromptProcessor.get_longest_prompt_length()
    if longest_prompts:
        report_dict["longest_prompts"] = longest_prompts
    report_dict["longest_prompt_length"] = PromptProcessor.get_longest_prompt_length()

    del report_dict["openai_key"]

    with open(Config.execution_report_file, "w") as json_file:
        json.dump(report_dict, json_file, indent=4)

    print("\n-------------------error messages-------------------")
    print(df_notes, end="\n\n")

    print("-----------------time-----------------")
    print(f"time: {time_:.2f}s")


def main(start_from_cp_file, overwrite):
    ensure_dataset_file_exists()

    df_checkpoint = None
    checkpoint_length = 0

    if start_from_cp_file is not False and os.path.isfile(
        Config.checkpoint_results_file
    ):
        if request_user_confirmation(
            f"Found checkpoint file {Config.checkpoint_results_file}. {Color.YELLOW}Append?{Color.END} [Y/n]: ",
            "Appending...",
            "Ignored checkpoint file.",
            auto_confirm="y" if start_from_cp_file else None,
        ):
            df_checkpoint = pd.read_csv(
                Config.checkpoint_results_file,
                dtype={"inference_results": "object", "notes": "object"},
            )
            index = df_checkpoint["inference_results"].first_valid_index()

            if index is not None:
                # the index of the first empty value after the first valid value
                for i in range(index, len(df_checkpoint["inference_results"])):
                    if pd.isna(df_checkpoint["inference_results"][i]):
                        checkpoint_length = i
                        break
                else:
                    checkpoint_length = len(df_checkpoint)
            else:
                # empty column "inference_results"
                checkpoint_length = 0

            print(f"Checkpoint length: {checkpoint_length}")

    if df_checkpoint is not None and checkpoint_length > 0:
        df = df_checkpoint.iloc[checkpoint_length:]
    else:
        df = loadTweets()

    if overwrite is None and os.path.exists(Config.results_file):
        print(f"{Config.results_file} already exists.\n")
        if not request_user_confirmation(
            f"{Color.YELLOW}Overwrite?{Color.END} [Y/n]: ",
            "Overwriting...",
            f"{Config.results_file} is skipped.",
            auto_confirm="y" if overwrite else None,
        ):
            exit(0)

    # if not overwrite:
    #     exit(0)

    df, df_with_not_empty_notes = predict(df, df_checkpoint, checkpoint_length)

    t2 = time.time()

    save_to_json(
        df_with_not_empty_notes.loc[:, ["id", "inference_results", "notes"]], t2 - t1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction model.")
    parser.add_argument(
        "-start_from_cp_file",
        action="store_true",
        help="Start from checkpoint file if exists.",
    )
    parser.add_argument(
        "-no_start_from_cp_file",
        action="store_false",
        dest="start_from_cp_file",
        help="Do not start from checkpoint file.",
    )
    parser.set_defaults(start_from_cp_file=None)
    parser.add_argument(
        "-overwrite",
        action="store_true",
        help="Overwrite existing results file.",
    )
    parser.add_argument(
        "-no_overwrite",
        action="store_false",
        dest="overwrite",
        help="Do not overwrite existing results file.",
    )
    parser.set_defaults(overwrite=None)
    args = parser.parse_args()

    start_from_cp_file = args.start_from_cp_file
    overwrite = args.overwrite
    
    # Set start method for multiprocessing to avoid CUDA initialization issues
    multiprocessing.set_start_method('spawn', force=True)
    
    # warnings.simplefilter('always', UserWarning)
    # warnings.filterwarnings('ignore')
    # warnings.filters.append(custom_warning_filter("A decoder-only architecture is being used, but right-padding was detected!"))
    # warnings.filters.append(custom_warning_filter("/home/syang662/miniconda3/envs/llm/lib/python3.12/site-packages/numpy/core/fromnumeric.py:59: FutureWarning: 'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead."))

    t1 = time.time()
    main(start_from_cp_file, overwrite)
