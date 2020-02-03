import configparser
import datetime
import json
import os
import pathlib
import pickle
import sys

from abc import ABC, abstractclassmethod, abstractproperty

import diskcache as dc
import numpy
import pandas as pd
import tensorflow as tf
import torch 
from transformers import *



def test_code():
    file_path = '/Users/ns5kn/Documents/insight/projects/factcc/models/saved_models/29-csvfile_5batch_7epoch/'

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    foo = BertForSequenceClassification.from_pretrained(file_path, from_tf=True)

    sent1= "what is happening"
    sent2= "what is NOT happening"
    encoded_input = tokenizer.encode_plus(
        sent1, 
        sent2, 
        add_special_tokens=True, 
        max_length=512,
        return_tensors='pt'
    )
        
    pred = foo(
        encoded_input['input_ids'], 
        token_type_ids=encoded_input['token_type_ids']
    )[0].argmax().item()
    return pred


print(test_code())