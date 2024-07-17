# File: parameters.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import os
import torch as th

# 04_29
PROMPTS = (
    "random_12_basic",
    "random_12_detailed",
    "random_15_basic",
    "random_15_detailed",
    "random_18_basic",
    "random_18_detailed",
    "random_21_basic",
    "random_21_detailed",
    "random_24_basic",
    "random_24_detailed",
    "random_27_basic",
    "random_27_detailed",
    "random_30_basic",
    "random_30_detailed",
    "random_3_basic",
    "random_3_detailed",
    "random_6_basic",
    "random_6_detailed",
    "random_9_basic",
    "random_9_detailed",
    "stratified_12_basic",
    "stratified_12_detailed",
    "stratified_15_basic",
    "stratified_15_detailed",
    "stratified_18_basic",
    "stratified_18_detailed",
    "stratified_21_basic",
    "stratified_21_detailed",
    "stratified_24_basic",
    "stratified_24_detailed",
    "stratified_27_basic",
    "stratified_27_detailed",
    "stratified_30_basic",
    "stratified_30_detailed",
    "stratified_3_basic",
    "stratified_3_detailed",
    "stratified_6_basic",
    "stratified_6_detailed",
    "stratified_9_basic",
    "stratified_9_detailed",
    "zero_0_basic",
    "zero_0_detailed",
)

# --------------------modify below------------------------
# --------------------------------------------------------
# --- WARNING: EXTREMELY DANGEROUS CODE BELOW ---
IS_TEST_RUN = False
# IS_TEST_RUN = True
# --- WARNING: EXTREMELY DANGEROUS CODE ABOVE ---

# MODEL_ABBR = "FLAN_UL2"
# MODEL = "google/flan-ul2"

# MODEL_ABBR = "GPT4"
# MODEL = "gpt-4-0125-preview"

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
# MODEL_ABBR = "MISTRAL_7B_INST"
# MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_ABBR = os.path.join(MODEL_ABBR, "with_special_tokens")
# MODEL_ABBR = os.path.join(MODEL_ABBR, "without_special_tokens")

# https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
MODEL_ABBR = "MISTRAL_8x7B_INST"
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# https://huggingface.co/mistralai/Mistral-7B-v0.3
# MODEL_ABBR = "MISTRAL_7B_BASE"
# MODEL = "mistralai/Mi/stral-7B-v0.3"

# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# MODEL_ABBR = "LLAMA3_8B_INST"
# MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_ABBR = os.path.join(MODEL_ABBR, "with_special_tokens")
# MODEL_ABBR = os.path.join(MODEL_ABBR, "without_special_tokens")

# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# MODEL_ABBR = "LLAMA3_70B_INST"
# MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"

# https://huggingface.co/meta-llama/Meta-Llama-3-8B
# MODEL_ABBR = "LLAMA3_8B_BASE"
# MODEL = "meta-llama/Meta-Llama-3-8B"

BATCH_SIZE = (
    1  # Preferably, set this to 1 if using Mistral. Idk why it's so slow on batches
)
MAX_OUTPUT_TOKENS = (
    200  # for mistral since it sometimes writes meaningless continuation
)
# LOAD_IN_8BIT = True
LOAD_IN_8BIT = None
LOAD_IN_4BIT = True
# LOAD_IN_4BIT = None

assert not (LOAD_IN_8BIT and LOAD_IN_4BIT)

# to exhaust gpu memory:
# flan-ul2 = 40GB, so run 2 instances
# mistral = 20GB, so run 4 instances
# llama3 = 20GB, so run 4 instances
MAX_CONCURRENT_API_CALLS = 2

DATE = "04_29"

# --- WARNING: EXTREMELY DANGEROUS CODE BELOW ---
REINFER = False
# REINFER = True
# --- WARNING: EXTREMELY DANGEROUS CODE ABOVE ---
ADD_PROMPT_PREFIX = False
# ADD_PROMPT_PREFIX = True

assert (REINFER and ADD_PROMPT_PREFIX) or (not REINFER and not ADD_PROMPT_PREFIX)

# --------------------modify above------------------------
# --------------------------------------------------------

COMPLETE_DATASET_FILENAME = f"all_prompts_{DATE}.csv"
TEST_DATASET_FILENAME = f"test_prompts_{DATE}.csv"

COMPLETE_RESULTS_FILENAME = f"all_results_{DATE}.csv"
TEST_RESULTS_FILENAME = f"test_results_{DATE}.csv"
COMPLETE_WARNINGS_FILENAME = f"all_warnings_{DATE}.csv"
TEST_WARNINGS_FILENAME = f"test_warnings_{DATE}.csv"

CHECKPOINT_RESULTS_FILENAME = f"cp_results_{DATE}.csv"
CHECKPOINT_WARNINGS_FILENAME = f"cp_warnings_{DATE}.csv"

EXECUTION_REPORT_FILENAME = f"execution_report_{DATE}.json"
FORMAT_WARNINGS_FILENAME = f"format_warnings_{DATE}.csv"
ANNOTATED_RESULTS_FILENAME = f"annotated_results_{DATE}.csv"
COMPLETION_FORMAT_ANALYSIS_FILENAME = f"completion_format_analysis_{DATE}.csv"

CM_DIR = f"cm_{DATE}"  # folder for classification matrices
CR_DIR = f"cr_{DATE}"  # folder for classification reports

DATASET_DIR = "dataset"
RESULTS_DIR = "results"
SUPERVISOR = "Varsha"
METRICS_DIR = "metrics"

COMPLETE_DATASET_FILE = os.path.join(DATASET_DIR, SUPERVISOR, COMPLETE_DATASET_FILENAME)
TEST_DATASET_FILE = os.path.join(DATASET_DIR, SUPERVISOR, TEST_DATASET_FILENAME)

COMPLETE_RESULTS_FILE = os.path.join(
    RESULTS_DIR, SUPERVISOR, MODEL_ABBR, COMPLETE_RESULTS_FILENAME
)
TEST_RESULTS_FILE = os.path.join(
    RESULTS_DIR, SUPERVISOR, MODEL_ABBR, TEST_RESULTS_FILENAME
)

COMPLETE_WARNINGS_FILE = os.path.join(
    RESULTS_DIR, SUPERVISOR, MODEL_ABBR, COMPLETE_WARNINGS_FILENAME
)
TEST_WARNINGS_FILE = os.path.join(
    RESULTS_DIR, SUPERVISOR, MODEL_ABBR, TEST_WARNINGS_FILENAME
)

CHECKPOINT_RESULTS_FILE = os.path.join(
    RESULTS_DIR, SUPERVISOR, MODEL_ABBR, CHECKPOINT_RESULTS_FILENAME
)
CHECKPOINT_WARNINGS_FILE = os.path.join(
    RESULTS_DIR, SUPERVISOR, MODEL_ABBR, CHECKPOINT_WARNINGS_FILENAME
)

EXECUTION_REPORT_FILE = os.path.join(
    RESULTS_DIR, SUPERVISOR, MODEL_ABBR, EXECUTION_REPORT_FILENAME
)

FORMAT_WARNINGS_FILE = os.path.join(
    RESULTS_DIR, SUPERVISOR, MODEL_ABBR, FORMAT_WARNINGS_FILENAME
)

ANNOTATED_RESULTS_FILE = os.path.join(
    RESULTS_DIR, SUPERVISOR, MODEL_ABBR, ANNOTATED_RESULTS_FILENAME
)

CM_FOLDER = os.path.join(RESULTS_DIR, SUPERVISOR, MODEL_ABBR, METRICS_DIR, CM_DIR)

CR_FOLDER = os.path.join(RESULTS_DIR, SUPERVISOR, MODEL_ABBR, METRICS_DIR, CR_DIR)

COMPLETION_FORMAT_ANALYSIS_FILE = os.path.join(
    RESULTS_DIR,
    SUPERVISOR,
    MODEL_ABBR,
    METRICS_DIR,
    COMPLETION_FORMAT_ANALYSIS_FILENAME,
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
    completion_format_analysis_file = COMPLETION_FORMAT_ANALYSIS_FILE
    model = MODEL
    model_abbr = MODEL_ABBR
    batch_size = BATCH_SIZE
    temperature = 1e-5
    top_p = 1
    patterns = [
        ["in favor", "in-favor"],
        ["against"],
        ["neutral or unclear", "unclear or neutral", "neutral"],
        ["not inferred"],
    ]
    # re_pattern = r"\b(in favor|against|neutral or unclear|unclear or neutral|not inferred)\b"
    # re_pattern = r"\b(" + "|".join(pattern) + r")\b"
    max_output_tokens = MAX_OUTPUT_TOKENS
    test_size = len(PROMPTS)
    max_concurrent_api_calls = MAX_CONCURRENT_API_CALLS
    # this is for gpt4 turbo (gpt-4-0125-preview). Modify if otherwise.
    cost_per_token = 1e-5  # https://openai.com/pricing
    total_cost = 0
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    prompt_version_list = PROMPTS
    load_in_8bit = LOAD_IN_8BIT
    load_in_4bit = LOAD_IN_4BIT

    reinfer = REINFER

    if reinfer:
        dataset_file = f"results/Varsha/{MODEL_ABBR}/all_dataset_{DATE}_reinfer.csv"
        results_file = f"results/Varsha/{MODEL_ABBR}/all_results_{DATE}_reinfer.csv"
        warnings_file = f"results/Varsha/{MODEL_ABBR}/all_warnings_{DATE}_reinfer.csv"
        execution_report_file = f"results/Varsha/{MODEL_ABBR}/execution_report_{DATE}_reinfer.json"
        format_warnings_file = f"results/Varsha/{MODEL_ABBR}/format_warnings_{DATE}_reinfer.csv"

    add_prompt_prefix = ADD_PROMPT_PREFIX
    

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python parameters.py <config_attribute>")
        sys.exit(1)

    attribute = sys.argv[1]
    config = Config()

    if hasattr(config, attribute):
        print(getattr(config, attribute))
    else:
        print(f"Unknown attribute: {attribute}")
        sys.exit(1)
