# File: parameters.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import os
import torch as th

PROMPTS = (
    "zero_0_basic", "zero_0_detailed", 
    "random_30_basic", "random_30_detailed", 
    "random_3_basic", "random_3_detailed", 
    "random_15_basic", "random_15_detailed", 
    "stratified_3_basic", "stratified_3_detailed", 
    "stratified_15_basic", "stratified_15_detailed", 
    "stratified_30_basic", "stratified_30_detailed", 
)

# --------------------modify these------------------------
# --------------------------------------------------------
IS_TEST_RUN = False

# MODEL_ABBR = "FLAN_UL2"
# MODEL = "flan-ul2"

MODEL_ABBR = "GPT4"
MODEL = "gpt-4-0125-preview"

# MODEL_ABBR = "MISTRAL"
# MODEL = "Mistral-7B-Instruct-v0.2"

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

EXCLUDED_PROMPTS = {
    "flan-ul2": (PROMPTS[2], PROMPTS[3], PROMPTS[12], PROMPTS[13]),
    "gpt-4-0125-preview": [],
    "Mistral-7B-Instruct-v0.2": []
}

BATCH_SIZE = 20
MAX_OUTPUT_TOKENS = 1000
MAX_CONCURRENT_API_CALLS = 16

# --------------------------------------------------------

COMPLETE_DATASET_FILENAME = "all_prompts_03_18.csv"
TEST_DATASET_FILENAME = "test_prompts_03_18.csv"

COMPLETE_RESULTS_FILENAME = "all_results_03_18.csv"
TEST_RESULTS_FILENAME = "test_results_03_18.csv"
COMPLETE_WARNINGS_FILENAME = "all_warnings_03_18.csv"
TEST_WARNINGS_FILENAME = "test_warnings_03_18.csv"

CHECKPOINT_RESULTS_FILENAME = "cp_results_03_18.csv"
CHECKPOINT_WARNINGS_FILENAME = "cp_warnings_03_18.csv"

EXECUTION_REPORT_FILENAME = "execution_report_03_18.json"
FORMAT_WARNINGS_FILENAME = "format_warnings_03_18.csv"
ANNOTATED_RESULTS_FILENAME = "annotated_results_03_18.csv"

CM_PARENT_PATH = "cm_03_18" # folder for classification matrices
CR_PARENT_PATH = "cr_03_18" # folder for classification reports

DATASET_PARENT_PATH = "dataset"
RESULTS_PARENT_PATH = "results"
SUPERVISOR = "Varsha"
METRICS_PARENT_PATH = "metrics"

COMPLETE_DATASET_FILE = os.path.join(
    DATASET_PARENT_PATH, SUPERVISOR, COMPLETE_DATASET_FILENAME
)
TEST_DATASET_FILE = os.path.join(
    DATASET_PARENT_PATH, SUPERVISOR, TEST_DATASET_FILENAME
)

COMPLETE_RESULTS_FILE = os.path.join(
    RESULTS_PARENT_PATH, SUPERVISOR, MODEL_ABBR, COMPLETE_RESULTS_FILENAME
)
TEST_RESULTS_FILE = os.path.join(
    RESULTS_PARENT_PATH, SUPERVISOR, MODEL_ABBR, TEST_RESULTS_FILENAME
)

COMPLETE_WARNINGS_FILE = os.path.join(
    RESULTS_PARENT_PATH, SUPERVISOR, MODEL_ABBR, COMPLETE_WARNINGS_FILENAME
)
TEST_WARNINGS_FILE = os.path.join(
    RESULTS_PARENT_PATH, SUPERVISOR, MODEL_ABBR, TEST_WARNINGS_FILENAME
)

CHECKPOINT_RESULTS_FILE = os.path.join(
    RESULTS_PARENT_PATH, SUPERVISOR, MODEL_ABBR, CHECKPOINT_RESULTS_FILENAME
)
CHECKPOINT_WARNINGS_FILE = os.path.join(
    RESULTS_PARENT_PATH, SUPERVISOR, MODEL_ABBR, CHECKPOINT_WARNINGS_FILENAME
)

EXECUTION_REPORT_FILE = os.path.join(
    RESULTS_PARENT_PATH, SUPERVISOR, MODEL_ABBR, EXECUTION_REPORT_FILENAME
)

FORMAT_WARNINGS_FILE = os.path.join(
    RESULTS_PARENT_PATH, SUPERVISOR, MODEL_ABBR, FORMAT_WARNINGS_FILENAME
)

ANNOTATED_RESULTS_FILE = os.path.join(
    RESULTS_PARENT_PATH, SUPERVISOR, MODEL_ABBR, ANNOTATED_RESULTS_FILENAME
)

CM_FOLDER = os.path.join(
    RESULTS_PARENT_PATH, SUPERVISOR, MODEL_ABBR, METRICS_PARENT_PATH, CM_PARENT_PATH
)

CR_FOLDER = os.path.join(
    RESULTS_PARENT_PATH, SUPERVISOR, MODEL_ABBR, METRICS_PARENT_PATH, CR_PARENT_PATH
)

class Config:
    openai_key = os.getenv("OPENAI_API_KEY")
    is_test_run = IS_TEST_RUN
    complete_dataset_file = COMPLETE_DATASET_FILE
    test_dataset_file = TEST_DATASET_FILE
    dataset_file = TEST_DATASET_FILE if is_test_run else COMPLETE_DATASET_FILE
    results_file = TEST_RESULTS_FILE if is_test_run else COMPLETE_RESULTS_FILE
    warnings_file = TEST_WARNINGS_FILE if is_test_run else COMPLETE_WARNINGS_FILE
    checkpoint_results_file = CHECKPOINT_RESULTS_FILE
    checkpoint_warnings_file = CHECKPOINT_WARNINGS_FILE
    execution_report_file = EXECUTION_REPORT_FILE
    format_warnings_file = FORMAT_WARNINGS_FILE
    annotated_results_file = ANNOTATED_RESULTS_FILE
    cm_folder = CM_FOLDER
    cr_folder = CR_FOLDER
    model = MODEL
    model_abbr = MODEL_ABBR
    batch_size = BATCH_SIZE
    temperature = 0
    pattern = ("in favor", "against", "neutral or unclear", "unclear or neutral", "not inferenced")
    # re_pattern = r"\b(in favor|against|neutral or unclear|unclear or neutral|not inferenced)\b"
    re_pattern = r"\b(" + "|".join(pattern) + r")\b"
    max_output_tokens = MAX_OUTPUT_TOKENS
    test_size = 14
    max_concurrent_api_calls = MAX_CONCURRENT_API_CALLS
    # this is for gpt4 turbo (gpt-4-0125-preview). Modify if otherwise.
    cost_per_token = 1e-5 # https://openai.com/pricing
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    excluded_prompts = EXCLUDED_PROMPTS
            
        

# ---------------------------------------------------------------
# - https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
# ---------------------------------------------------------------

# --- gpt-4 turbo
# MODEL = "gpt-4-0125-preview"

# --- gpt-chat-4
# MODEL = "gpt_chat_4"

# --- gpt-chat-turbo-3_5
# MODEL = "gpt-3.5-turbo-0613"
# MODEL = "gpt_chat_turbo_3_5"

# --- flan
# MODEL = "flan-t5-large"
# MODEL = "flan-t5-xxl"
# MODEL = "flan-ul2"
