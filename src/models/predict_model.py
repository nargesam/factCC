import configparser
import datetime
import json
import os
import pathlib
import pickle

import numpy
import pandas as pd
import tensorflow as tf
import torch 

from transformers import *
# Load the TensorFlow model in PyTorch for inspection
modelpath = os.path.join(PROJECT_DIR, 'models/saved_models/test/')
model.save_pretrained(modelpath)


# Load and test the saved model
respath = os.path.join(PROJECT_DIR)
pytorch_model = BertForSequenceClassification.from_pretrained(respath, from_tf=True)
# print("model done")


# Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
sentence_0 = "This research was consistent with his findings."
sentence_1 = "His findings were compatible with this research."
sentence_2 = "His findings were not compatible with this research."
inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')

pred_1 = pytorch_model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
# 
pred_2 = pytorch_model(inputs_2['input_ids'], token_type_ids=inputs_2['token_type_ids'])[0].argmax().item()

print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")

