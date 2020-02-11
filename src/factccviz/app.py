import os
import sys

import pandas as pd
import streamlit as st

from transformers import BertTokenizer

import lib
import util

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = lib.load_dataset()
dataset = dataset.query("label == 'INCORRECT'")
# row_pos = 0

st.title('FactCCViz')

selected_instance = st.sidebar.selectbox(
    'Which dataset?',
     list(range(30))
)



st.header("Dataset")
st.write(dataset)

st.header("Claim")
claim_1 = st.empty()
claim_2 = st.empty()
claim_3 = st.empty()

# st.header("Source")
src_full = dataset.iloc[selected_instance, -1]
src_full_tokenized = tokenizer.tokenize(src_full)

# Extraction span
extraction_span = dataset.iloc[selected_instance, 3]
i, j = extraction_span
src_full_tokenized[i] = "```" + src_full_tokenized[i]
src_full_tokenized[j] = "```" + src_full_tokenized[j]

checkbox = st.sidebar.checkbox('See source')
if checkbox:
    st.markdown(" ".join(src_full_tokenized))



claim = dataset.iloc[selected_instance, 4]
claim_tokenized = tokenizer.tokenize(claim)

# Augmentation span
augmentation_span = dataset.iloc[selected_instance, 6]
i, j = augmentation_span
claim_tokenized[i] = "```" + claim_tokenized[i]
claim_tokenized[j+1] = "```" + claim_tokenized[j+1]

augmentation_type = str(dataset.iloc[selected_instance, 5])
claim_1.markdown(f"**Augmentation type**: {augmentation_type}")
claim_2.markdown(f"**Augmentation span**: ({i}, {j})")

claim = " ".join(claim_tokenized)
claim_3.markdown(f"**Augmentation**: {claim}")
