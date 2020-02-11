import configparser
import datetime
import json
import os
import pathlib
import pickle
import sys

from abc import ABC, abstractclassmethod, abstractproperty

import diskcache as dc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch 
from transformers import *


# Must add this environment variable or everything explodes
# HT: https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# When set to -1 you use no GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


class FactCCViz():
    def __init__(self, config_path):
        self._config_path = config_path
        self._load_config()
        # self._create_output_directory()
    
    def _get_config(self):
        config = configparser.RawConfigParser()
        try:
            config.read(self._config_path)
            return config
        except:
            print("Couldn't read config file from ./configuration.cfg")
            exit()
    
    def _load_config(self):
        config = self._get_config()

        # self._dataset_jsonl_path = config.get('Params', 'dataset')
        # self._dataset_csv_path = config.get('Params', 'csvfile')
        self._test_csv_path = config.get('Params', 'testcsvfile')
        self._cache_dir = config.get('Params', 'cache_dir')
        self._tf_cache_dir = config.get('Params', 'tf_cache_dir')
        self._save_model_dir = config.get('Params', 'save_model_dir')
        # self._datasize = config.get('Params', 'datasize')
        # self._token_maxlen = int(config.get('Params', 'maxlen'))
        # self._batch_size = int(config.get('Params', 'batchsize'))
        # self._steps = int(config.get('Params', 'steps'))
        # self._num_epochs = int(config.get('Params', 'num_epochs'))
    
    def load_data(self, clobber=True):
        """ if Clobber= False, use the cached value if it exists, if Clobber==True, always read the data
         Read Json or CSV file depend on your choice """
        key = "FULL_DATASET"

        with dc.Cache(self._cache_dir) as cache:
            if clobber and key in cache:
                del cache[key]

            if key in cache:
                self._dataset = cache.get(key)
                # print("cache used")
            else:
                # print("cache not used")
                self._dataset = self._read_csv()
                cache.set(key, self._dataset)

        
    def _read_csv(self):
        data = pd.read_csv(self._test_csv_path)
        # data =  data.rename(columns={'id': 'idx', 'text': 'sentence1', 'claim': 'sentence2'})
        # data = data[['idx', 'sentence1', 'sentence2', 'label']]
        return data
    
    def augmentation_plot(self):

        self.dataset["label"] = self.dataset["label"].map({'SUPPORTS': numpy.int64(1) ,'REFUTES': numpy.int64(0)})
        plot = self.dataset[self.dataset['label'] != self.dataset['pred']]
        # plot['augmentation'].value_counts()
        fig, ax = plt.subplots()
        ax.set_title('Comparison of different augmentation fact that the summarization model could not detect')

        ax.set_xlabel('Augmentation method')
        ax.set_ylabel('Count')
        # data =  obj.dataset[obj.dataset[]]
        plot['augmentation'].value_counts().plot(ax=ax, kind='bar')
        plt.savefig(self.save_plot_augmentation_path)
    

    def _format_attention(self, attention):
        squeezed = []
        for layer_attention in attention:
            squeezed.append(layer_attention.squeeze(0))
        return torch.stack(squeezed)


    def heatmap_plot(self, sentence_a, sentence_b, layer=10, head=0):
        self._layer = layer
        self._head = head 

        inputs = self.tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)

        token_type_ids = inputs['token_type_ids']
        input_ids = inputs['input_ids']
        input_id_list = input_ids[0].tolist() # Batch index 0

        attention = self.transformer(input_ids, token_type_ids=token_type_ids)[-1]
        att = self._format_attention(attention)
        last_last = att[layer,head,:,:]
        # second_first = att[2,0,:,:]

        tokens = self.tokenizer.convert_ids_to_tokens(input_id_list)

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


    @property
    def dataset(self):
        return self._dataset
    
    @property
    def save_model_dir(self):
        return self._save_model_dir

    @property
    def save_plot_augmentation_path(self):
        return os.path.join(self.save_model_dir, 'augmentation.png')
    
    @property
    def save_plot_heatmap_path(self):
        res_name = f"heatmap-layer-{self._batch_size}__epoch-{self._num_epochs}__datasize-{self._datasize}"
        self._output_dir = os.path.join(self._save_model_dir, self.model_type, res_name)
        return os.path.join(self.save_model_dir, 'heat.png')

    @property
    def model_type(self):
        # return self._model_type
        return "bert-base-uncased"

    @property
    def tokenizer(self):
        return self._get_tokenizer()
    
    def _get_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_type, do_lower_case=True)
        return tokenizer

    @property
    def transformer(self):
        return self._get_transformer()
    
    def _get_transformer(self):
        model = BertModel.from_pretrained(self.model_type, output_attentions=True)
        return model

    # def load_transformer(self):
    #     self._transformer = self._get_transformer()

    # def load_tokenizer(self):
    #     self._tokenizer =  self._get_tokenizer()
