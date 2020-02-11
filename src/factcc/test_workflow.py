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

from model import FactCCBertBaseCased, FactCCBertBaseUncased
from visualization import FactCCViz

def load_model_from_file(output_dir):
    model = BertForSequenceClassification.from_pretrained(output_dir, from_tf=True)
    return model


def load_test_data(test_csv_path):
        test_data = pd.read_csv(test_csv_path)
        test_data =  test_data.rename(columns={'id': 'idx', 'text': 'sentence1', 'claim': 'sentence2'})
        # print(len(test_data))
        # print(test_data.columns)

        return test_data


def run_test(model_path, testdata_path):
    
    model = load_model_from_file(model_path)
    test_data = load_test_data(testdata_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    data_dict = {}
    data_dict['idx'] = []
    data_dict['pred'] = []


    # i = 0
    for index, row in test_data.iterrows():

        encoded_input = tokenizer.encode_plus(
            row['sentence1'], 
            row['sentence2'], 
            max_length=512,
            add_special_tokens=True, 
            return_tensors='pt'
        )
        # print(encoded_input)
        pred = model(
            encoded_input['input_ids'], 
            token_type_ids=encoded_input['token_type_ids']
        )[0].argmax().item()
        
        data_dict['idx'].append(str(row['idx']))
        data_dict['pred'].append(pred)
        # print(data_dict)
        # i += 1
        # if i ==5:
        #     break

    df_pred = pd.DataFrame(data_dict)
    # print(len(df_pred))
    test_with_predictions = pd.merge(test_data, df_pred, on='idx', how='inner')
    return test_with_predictions

def save_test(test_with_predictions, save_path):
    test_with_predictions.to_csv(save_path)
    



def augmentation_plot(dataset, save_plot_augmentation_path):

    dataset["label"] = dataset["label"].map({'SUPPORTS': numpy.int64(1) ,'REFUTES': numpy.int64(0)})
    plot = dataset[dataset['label'] != dataset['pred']]
    # plot['augmentation'].value_counts()
    fig, ax = plt.subplots()
    ax.set_title('Comparison of different augmentation fact that the summarization model could not detect')

    ax.set_xlabel('Augmentation method')
    ax.set_ylabel('Count')
    # data =  obj.dataset[obj.dataset[]]
    plot['augmentation'].value_counts().plot(ax=ax, kind='bar')
    plt.savefig(save_plot_augmentation_path)
    

def _format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        squeezed.append(layer_attention.squeeze(0))
    return torch.stack(squeezed)


def heatmap_plot( sentence_a, sentence_b, layer=10, head=0):

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-base-cased', output_attentions=True)

    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)

    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    input_id_list = input_ids[0].tolist() # Batch index 0

    attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    att = _format_attention(attention)
    last_last = att[layer,head,:,:]
    # second_first = att[2,0,:,:]

    tokens = tokenizer.convert_ids_to_tokens(input_id_list)

    sentence_b_start = token_type_ids[0].tolist().index(1)
    len_target_token = len(tokens) - sentence_b_start
    len_source_token = sentence_b_start
    a_tokens = tokens[:sentence_b_start]
    b_tokens = tokens[sentence_b_start:]

    last_last_ab = last_last[:sentence_b_start:, sentence_b_start:len(tokens)]
    last_last_ab = last_last_ab.detach().numpy()

    fig = go.Figure(data=go.Heatmap(
            z=last_last_ab,
            x=b_tokens,
            y=a_tokens,
            colorscale='PuBuGn')      
            )

    fig.update_layout(
        title='Text vs Claim',
        xaxis_nticks=36)

    fig.show()
    fig.write_image("images/fig1.png")







if __name__ == "__main__":
    # pred_config_path = '/Users/ns5kn/Documents/insight/projects/factCC/src/factcc/scripts/test_workflow.cfg'
    # viz_config_path = '/Users/ns5kn/Documents/insight/projects/factCC/src/factcc/scripts/test_workflow.cfg'
    # load_config(config_path)
    """ Read the model, read the file to predict, save it to the address to be read by FactCCViz """
    # obj_pred = FactCCBertBaseCased(pred_config_path)
    # obj_pred.load_model_from_file()
    # obj_pred.load_test_data()
    # obj_pred.run_test()
    # obj_pred.save_test()
    load_model_path = '/Users/ns5kn/Documents/insight/projects/factCC/models/saved_models/bert-base-cased/batch-50__epoch-6__datasize-10perc__steps-8__acc-73/'
    testdata_to_predict_path = '/Users/ns5kn/Documents/insight/projects/factCC/data/interim/sample_train_1k_fromclipped_test.csv'
    pred_save_path = '/Users/ns5kn/Documents/insight/projects/factCC/data/interim/prediction_on_test.csv'

    test_with_predictions = run_test(load_model_path, testdata_to_predict_path)

    save_test(test_with_predictions, pred_save_path)

    pred_read_path = '/Users/ns5kn/Documents/insight/projects/factCC/data/interim/prediction_on_test.csv'


    # """ This will read in your predicted text and plots heatmap and frequency based on your predicted labels"""
    # obj = FactCCViz(config_path)
    # data_to_plot = obj.load_data
    # obj.augmentation_plot()
    # sent = '2'
    # sent2 = '3'
    # obj.heatmap_plot(sent, sent2)

