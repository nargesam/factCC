import os
import sys

import pandas as pd
import streamlit as st
import tensorflow as tf
import torch 

from transformers import *

import lib
import util

dataset = lib.load_dataset()
row_pos = 19

st.title('FactCCViz')

st.header("Source")

span = dataset.iloc[row_pos, 3]

src_full = dataset.iloc[row_pos, -1]

tokenizer_mode = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_mode)
src_full_tokenized = tokenizer.tokenize(src_full)

src_full_tokenized[span[0]] = "```" + src_full_tokenized[span[0]]
src_full_tokenized[span[1]] = src_full_tokenized[span[1]] + "```"

st.write(" ".join(src_full_tokenized))

# st.markdown(src_full)

# st.header("Extract from source")
# # # TODO: extract start and end of span with regex
# # span = span.replace("[", "")
# # span = span.replace("]", "")
# # span = span.split(", ")
# # span_start = int(span[0].strip())
# # span_end = int(span[1].strip())


# # src_extract = src_full.split(" ")
# src_extract = src_full_tokenized[span[0]: span[1]+1]


# st.markdown(" ".join(src_extract))

# st.write(span)

st.header("Claim")
claim = dataset.iloc[row_pos, 4]
claim_tokenized = tokenizer.tokenize(claim)


span = dataset.iloc[row_pos, 6]
claim_tokenized[span[0]] = "```" + claim_tokenized[span[0]]
claim_tokenized[span[1]] = claim_tokenized[span[1]] + "```"

st.markdown(" ".join(claim_tokenized))
