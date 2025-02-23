# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import random
import os
import numpy as np
import gc
import re
import struct
from tqdm import tqdm

SOURCE_DATA_PATH = "path/to/source/data.txt"
os.makedirs(os.path.dirname("data-dp/"), exist_ok=True)

# Logging boilerplate
log_file = open("data_prep.log", "w")
pbar_recept_string = " " * 200 + "\n"
log_file.write(pbar_recept_string)
log_file.write(pbar_recept_string)
log_file.flush()
def log(s:str, p_level=None):
    if p_level == 1:
        log_file.seek(0,0)
        log_file.write(pbar_recept_string)
        log_file.seek(0,0)
        log_file.write(s)
        log_file.seek(0,2)
    elif p_level == 2:
        log_file.seek(len(pbar_recept_string), 0)
        log_file.write(pbar_recept_string)
        log_file.seek(len(pbar_recept_string), 0)
        log_file.write(s)
        log_file.seek(0,2)
    else:
        if s[0].upper() == s[0]:
            start = "\n"
            end = ":"
        else:
            start = "    --> "
            end = ""
        log_file.write(start + s + end + "\n")
    log_file.flush()

## Convert seconds to days, hours, minutes, seconds
def convert_seconds(seconds:float):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (days, hours, minutes, seconds)

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

class TinypyTokenizer():
    def __init__(self):
        # defining the keywords list
        self.keywords = sorted([
            '# ', '&', ';', '?', '|', '@', '^', '$',
            '# output\n', '# code\n', '\n#STEP\n',
            '\n', '\t', '\n\n',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'while', '=', '+',
            '(', ')', ':', ',', '.',
            'if', 'elif', 'else',
            '==', '<', '>', 'not', '!=',
            'print(',
        ], key = len, reverse = True)
        pattern = "|".join([re.escape(kw) for kw in self.keywords]) + '|[^ ]+?'
        self.regex = re.compile(pattern)
        self.encod_map = { kw : i for i, kw in enumerate(self.keywords)}
        self.decod_map = { i : kw for i, kw in enumerate(self.keywords)}

    def tokenize(self, input_string):
        return self.regex.findall(input_string)

    def encode(self, input_string):
        tokens = self.tokenize(input_string)
        return [self.encod_map[token] for token in tokens]

    def decode(self, tokens_ids):
        return [self.decod_map[id] for id in tokens_ids]

# Load the dataset
log("Loading the dataset")
with open(SOURCE_DATA_PATH, "r") as f:
    data = f.read()

# Split by examples using \n\n
log("Splitting by \\n\\n")
examples = data.split("\n\n")[:-1]
log(f"Total number of examples: {len(examples):,}")

# Introduce Execution Step IDs for Memory-Augmented Attention
log("Adding execution step IDs")
processed_examples = []
step_counter = 0

for example in examples:
    processed_examples.append(f"\n#STEP {step_counter}\n{example}")
    step_counter += 1

examples = processed_examples
del data
gc.collect()

# Creating train, val, test splits
log("Creating dataset splits")
train_examples = examples[:1_000_000]
val_examples = examples[1_000_000:1_010_000]
test_examples = examples[1_010_000:1_010_100]

log(f"Train: {len(train_examples)} | Val: {len(val_examples)} | Test: {len(test_examples)}")

# Save text files
log("Saving text datasets")
with open("data-dp/train.txt", 'w') as f:
    f.write("\n\n".join(train_examples) + "\n\n")
with open("data-dp/val.txt", 'w') as f:
    f.write("\n\n".join(val_examples) + "\n\n")
with open("data-dp/test.txt", 'w') as f:
    f.write("\n\n".join(test_examples) + "\n\n")

del train_examples, val_examples, test_examples
gc.collect()

# Tokenization
log("Tokenizing datasets")
tpt = TinypyTokenizer()

def encode_to_memmap(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        examples_string = f.read()
    examples = examples_string.split('\n\n')[:-1]
    output_memmap = np.memmap(output_file_path, dtype=np.uint8, mode='w+', shape=(sum(len(tpt.encode(e)) for e in examples),))
    
    index = 0
    for example in tqdm(examples):
        tokenized_example = tpt.encode(example)
        output_memmap[index:index+len(tokenized_example)] = tokenized_example
        index += len(tokenized_example)
    
    del output_memmap
    gc.collect()
    return None

log("Encoding train.txt to train.bin")
encode_to_memmap("data-dp/train.txt", "data-dp/train.bin")

log("Encoding val.txt to val.bin")
encode_to_memmap("data-dp/val.txt", "data-dp/val.bin")

log("Saving vocab size")
with open("data-dp/vocab_size.txt", "w") as f:
    f.write(str(len(tpt.keywords)))

log_file.close()