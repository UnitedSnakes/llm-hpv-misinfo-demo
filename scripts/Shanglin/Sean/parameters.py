# File: parameters.py
# Author: Shanglin Yang
# Contact: syang662@wisc.edu

import os
import openai

# Obtain OPENAI key from env variables
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY

DATASET_ROOT_PATH = "dataset"
SUPERVISOR = "Varsha"
DATASET_UNDER_SUPERVISOR_PATH = os.path.join(DATASET_ROOT_PATH, SUPERVISOR)
# DATASET_MISINFO_OR_NOT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "misinfo_or_not")
# DATASET_ORIGINAL_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "original")

TEST = True
TEST_RESULTS_FOLDER = "test_results"
COMPLETE_RESULTS_FOLDER = "results"
# OUTPUT_FOLDER_PATH = "test_results"
OUTPUT_FOLDER_PATH = (
    os.path.join(TEST_RESULTS_FOLDER, SUPERVISOR)
    if TEST
    else os.path.join(COMPLETE_RESULTS_FOLDER, SUPERVISOR)
)
# OUTPUT_MISINFO_OR_NOT_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "misinfo_or_not")

# TSV_LIST = os.listdir(DATASET_MISINFO_OR_NOT_FOLDER_PATH)

# ---------------------------------------------------------------
# - https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
# ---------------------------------------------------------------

# --- gpt-4 turbo
MODEL = "gpt-4-0125-preview"
# --- gpt-chat-4
# MODEL = "gpt_chat_4"

# --- gpt-chat-turbo-3_5
# MODEL = "gpt-3.5-turbo-0613"
# MODEL = "gpt_chat_turbo_3_5"

# --- gpt_3_davinci
# MODEL = "gpt_3_davinci" # text-davinci-003

# --- gpt-2
# MODEL = "gpt_2"

# --- flan
# MODEL = "flan-t5-large"
# MODEL = "flan-t5-xxl"
# MODEL = "flan-ul2"

# Feed the model in batches to speed up the computation
# BATCH_SIZE = 8
BATCH_SIZE = 16
# BATCH_SIZE = 32


# PROMPT_DICT = {
#     "zero": PROMPT_ZERO,
#     "few_k2": PROMPT_FEW_K2,
#     "few_k4": PROMPT_FEW_K4,
#     "few_k6": PROMPT_FEW_K6,
# }

# FEW_SHOT_EXAMPLES_DICT = {
#     "few_k2": FEW_SHOT_EXAMPLES[:2],
#     "few_k4": FEW_SHOT_EXAMPLES[:4],
#     "few_k6": FEW_SHOT_EXAMPLES[:],
# }

PATTERN = r"\b(yes|no)\b"

TEMPERATURE = 0

MAX_OUTPUT_TOKENS = 10  # 5 ?

NUM_CORES_CALL_API = 20  # increase to 50 ??

# for version, prompt in PROMPT_DICT.items():
#     print(version)
#     print(prompt)

# pd.DataFrame(PROMPT_DICT.items(), columns=["version", "prompt"]).to_csv(
#     "scripts/Shanglin/few_shot_prompts.csv", index=False
# )
MAX_INPUT_TOKENS = 512
