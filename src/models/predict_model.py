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
# model.save_pretrained(modelpath)

model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')


# Load and test the saved model
# respath = os.path.join(PROJECT_DIR)

# PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
# modelpath = os.path.join(PROJECT_DIR, 'models/saved_models/test/')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
modelpath = '~/Documents/insight/projects/factCC/models/saved_models/'
pytorch_model = BertForSequenceClassification.from_pretrained(modelpath , from_tf=True)
# pytorch_model = BertForSequenceClassification.from_pretrained(modelpath, from_tf=True)

# # Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
sentence_0 = "This research was consistent with his findings."
sentence_1 = "His findings were compatible with this research."
# sentence_2 = "His findings were not compatible with this research."
inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
# inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')

pred_1 = pytorch_model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
# pred_2 = pytorch_model(inputs_2['input_ids'], token_type_ids=inputs_2['token_type_ids'])[0].argmax().item()


token_type_ids = inputs['token_type_ids']
input_ids = inputs['input_ids']
attention = pytorch_model(input_ids, token_type_ids=token_type_ids)[-1]

print(len(attention))
print(type(attention))
print(attention)




# print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
# print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")



# from bertviz import head_view
# from transformers import BertTokenizer, BertModel



# def show_head_view(model, tokenizer, sentence_a, sentence_b):

#     inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
#     token_type_ids = inputs['token_type_ids']
#     input_ids = inputs['input_ids']
#     attention = model(input_ids, token_type_ids=token_type_ids)[-1]
#     input_id_list = input_ids[0].tolist() # Batch index 0
#     tokens = tokenizer.convert_ids_to_tokens(input_id_list)
#     if sentence_b:
#         sentence_b_start = token_type_ids[0].tolist().index(1)
#     else:
#         sentence_b_start = None
#     head_view(attention, tokens, sentence_b_start)



# model_version = 'bert-base-uncased'
# do_lower_case = True
# # model = BertForSequenceClassification.from_pretrained(modelpath, from_tf=True)
# model = BertModel.from_pretrained(model_version, output_attentions=True)
# tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
# sentence_a = "The cat sat on the mat"
# sentence_b = "The cat lay on the rug"
# show_head_view(model, tokenizer, sentence_a, sentence_b)





def plot_attention(source, target, attention):
    # Creates a heatmap attention score plot based on source and target tokens
    # Source and target must be tokenized space delimited strings
    # source_toks = source.split(' ')
    # target_toks = target.split(' ')

    # Attention score is padded based on batch prediction
    # We truncate to the relevant size based on source and target tokens
    attention = attention[:len(target_toks), :len(source_toks)]
    figsize = (attention.shape[1]//4, attention.shape[0]//4)
    fig, ax1 = plt.subplots(figsize=figsize)

    ax = sns.heatmap(attention, linewidths=0.1, ax=ax1, linecolor='black',
                    xticklabels=source_toks, yticklabels=target_toks, square=True,
                    cmap=sns.color_palette("YlGn", n_colors=15), 
                    cbar_kws={"shrink": 0.5, 'pad':0.04, 'label': 'Attention Score'})

    ax.set_xlabel('Source Tokens')
    ax.set_ylabel('Prediction Tokens')

    loc, labels = plt.yticks()
    ax.set_yticklabels(labels, rotation=360)

    ax.tick_params(top=True, bottom=False,
                labeltop=False, labelbottom=True)

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    return ax
