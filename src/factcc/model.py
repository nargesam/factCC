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

# import util

# Must add this environment variable or everything explodes
# HT: https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# When set to -1 you use no GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

class FactCCBase(ABC):
    def __init__(self, config_path):
        self._config_path = config_path
        self._load_config()
        self._create_output_directory()
        
    @abstractproperty
    def model_type(self):
        return NotImplementedError()
    
    @abstractproperty
    def tokenizer(self):
        return NotImplementedError()
    
    @abstractproperty
    def transformer(self):
        return NotImplementedError()
    
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

        self._dataset_jsonl_path = config.get('Params', 'dataset')
        self._dataset_csv_path = config.get('Params', 'csvfile')
        self._test_csv_path = config.get('Params', 'testcsvfile')
        self._cache_dir = config.get('Params', 'cache_dir')
        self._tf_cache_dir = config.get('Params', 'tf_cache_dir')
        self._save_model_dir = config.get('Params', 'save_model_dir')
        self._datasize = config.get('Params', 'datasize')
        self._token_maxlen = int(config.get('Params', 'maxlen'))
        self._batch_size = int(config.get('Params', 'batchsize'))
        self._steps = int(config.get('Params', 'steps'))
        self._num_epochs = int(config.get('Params', 'num_epochs'))
    
    def load_data(self, clobber=False):
        """ if Clobber= False, use the cached value if it exists, otherwise, always read the data
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
        data = pd.read_csv(self._dataset_csv_path)
        data =  data.rename(columns={'id': 'idx', 'text': 'sentence1', 'claim': 'sentence2'})
        data = data[['idx', 'sentence1', 'sentence2', 'label']]
        return data
    
    def split_train_test(self, clobber=False):

        train_key = "TRAIN_DATASET"
        validation_key = "VALIDATION_DATASET"

        with dc.Cache(self._cache_dir) as cache:
            if clobber and train_key in cache and validation_key in cache:
                del cache[train_key]
                del cache[validation_key]

            if train_key in cache and validation_key in cache:
                self._train_dataset = cache.get(train_key)
                self._validation_dataset = cache.get(validation_key)
                print("cache used")
            else:
                print("cache not used")
                self._train_dataset = self._dataset.sample(frac=0.8, replace=False, random_state=1)
                self._validation_dataset = self._dataset[~self._dataset.isin(self._train_dataset)].dropna()

                self._train_dataset["label"] = self._train_dataset["label"].map(
                    {
                        'CORRECT': numpy.int64(1),
                        'INCORRECT': numpy.int64(0),
                        'SUPPORTS': numpy.int64(1),
                        'REFUTES': numpy.int64(0)
                    },
                )
                self._validation_dataset["label"] = self._validation_dataset["label"].map(
                    {
                        'CORRECT': numpy.int64(1),
                        'INCORRECT': numpy.int64(0),
                        'SUPPORTS': numpy.int64(1),
                        'REFUTES': numpy.int64(0)
                    },
                )

                cache.set(train_key, self._train_dataset)
                cache.set(validation_key, self._validation_dataset)

                # train["label"] = train["label"].map({'SUPPORTS': numpy.int64(1) ,'REFUTES': numpy.int64(0)})
                # validation["label"] = validation["label"].map({'CORRECT': numpy.int64(1) ,'INCORRECT': numpy.int64(0)})
                # validation["label"] = validation["label"].map({'CORRECT': numpy.int64(1) ,'INCORRECT': numpy.int64(0)})
    
    def convert_examples_to_features(self, clobber=False):
        # tokenizer = BertTokenizer.from_pretrained(tokenizer_mode)
        train_tfdata = tf.data.Dataset.from_tensor_slices(dict(self._train_dataset))
        validation_tfdata = tf.data.Dataset.from_tensor_slices(dict(self._validation_dataset))

        train_key = "TRAIN_FEATURES"
        validation_key = "VALIDATION_FEATURES"

        with dc.Cache(self._cache_dir) as cache:

            if clobber and train_key in cache and validation_key in cache:
                del cache[train_key]
                del cache[validation_key]  

            if train_key in cache and validation_key in cache:
                print("cache used")
                self._train_features = cache[train_key]
                self._validation_features = cache[validation_key]
            else:
                print("cache not used")
                self._train_features = glue_convert_examples_to_features(
                    train_tfdata, 
                    self.tokenizer, 
                    label_list=[numpy.int64(1), numpy.int64(0)], 
                    max_length=self._token_maxlen, 
                    task='mrpc', 
                    output_mode="classification"
                )
                
                self._validation_features = glue_convert_examples_to_features(
                    validation_tfdata, 
                    self.tokenizer, 
                    label_list=[numpy.int64(1), numpy.int64(0)], 
                    max_length=self._token_maxlen, 
                    task='mrpc', 
                    output_mode="classification"
                )

                # cache.set(train_key, self._train_features)
                # cache.set(validation_key, self._validation_features)
    
    def create_model(self):
        self._model = self.transformer
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self._model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


    def train_model(self):
        self._train_features = self._train_features.shuffle(100).batch(self._batch_size).repeat(2)
        self._validation_features = self._validation_features.batch(self._batch_size)
        
        # cp_callback = tf.keras.callbacks.ModelCheckpoint((filepath=checkpoint_path,
        #                                          save_weights_only=True,
        #                                          verbose=1)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_output_path, monitor='val_accuracy', verbose=1,
                         save_best_only=True, mode='max') #, save_freq='epoch')
        callbacks=[model_checkpoint]
        self._history = self._model.fit(self._train_features, epochs=self._num_epochs, steps_per_epoch=self._steps,
                        validation_data=self._validation_features, callbacks=callbacks) #, validation_steps=2)


    def _create_output_directory(self):
        current_time = datetime.datetime.now()
        # folder_name = str(self.) + "-" + str(current_time.hour) + "-" + str(current_time.minute)
        # model_type_dir = f"{self._model_type}"
        res_name = f"batch_size-{self._batch_size}__epoch-{self._num_epochs}__datasize-{self._datasize}"
        self._output_dir = os.path.join(self._save_model_dir, self.model_type, res_name)
        
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir) 

    def save_model(self):
        with open(self.history_output_path, 'wb') as f:
                pickle.dump(self._history.history, f)

        # print(self._output_dir)
        # Load the TensorFlow model in PyTorch for inspection
        # modelpath = os.path.join(PROJECT_DIR, 'models/saved_models/test/')
        self._model.save_pretrained(self._output_dir)
    

    def load_model_from_file(self):
        self._model = BertForSequenceClassification.from_pretrained(self.output_dir, from_tf=True)

    def load_test_data(self):
        self._test_data = pd.read_csv(self._test_csv_path)
        self._test_data =  self._test_data.rename(columns={'id': 'idx', 'text': 'sentence1', 'claim': 'sentence2'})
        # self._test_data = self._test_data[['idx', 'sentence1', 'sentence2', 'label']]

    def run_test(self):
        data_dict = {}
        data_dict['idx'] = []
        data_dict['pred'] = []

        # i = 0
        for index, row in self._test_data.iterrows():
            
            encoded_input = self.tokenizer.encode_plus(
                row['sentence1'], 
                row['sentence2'], 
                max_length=self._token_maxlen,
                add_special_tokens=True, 
                return_tensors='pt'
            )
            # print(encoded_input)
            pred = self._model(
                encoded_input['input_ids'], 
                token_type_ids=encoded_input['token_type_ids']
            )[0].argmax().item()
            
            data_dict['idx'].append(str(row['idx']))
            data_dict['pred'].append(pred)
            # print(self._data_dict)


        df_pred = pd.DataFrame(data_dict)
        self._test_with_predictions = pd.merge(self._test_data, df_pred, on='idx', how='inner')
    
    def save_test(self, save_path=None):
        if save_path is None:
            save_path = self.test_data_output_path
        
        self._test_with_predictions.to_csv(save_path)
    

    def model_exists(self):
        return os.path.isfile(self.model_output_path) and os.path.isfile(self.history_output_path)
    

    def test_pred_exists(self):
        return os.path.isfile(self.test_data_output_path) 


    @property
    def test_data_output_path(self):
        return os.path.join(self.output_dir, 'prediction_on_test.csv')


    @property
    def model_output_path(self):
        return os.path.join(self.output_dir, 'tf_model.h5')

    @property
    def output_dir(self):
        return self._output_dir
    
    @property
    def history_output_path(self):
        return os.path.join(self.output_dir, 'history.pickle')

    @property
    def dataset(self):
        return self._dataset

    @property
    def train_dataset(self):
        return self._train_dataset
    
    @property
    def validation_dataset(self):
        return self._validation_dataset
    
    @property
    def model(self):
        return self._model
    

class FactCCBertBaseCased(FactCCBase):
    def __init__(self, config_path):
        super().__init__(config_path)
        
        self._tokenizer = None
        self._transformer = None
        # self._model_type = "bert-base-cased"

    @property
    def model_type(self):
        # return self._model_type
        return "bert-base-cased"

    @property
    def tokenizer(self):
        return self._tokenizer
    
    def _get_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_type)
        return tokenizer

    def load_tokenizer(self):
        self._tokenizer =  self._get_tokenizer()

    @property
    def transformer(self):
        return self._transformer
    
    def _get_transformer(self):
        model = TFBertForSequenceClassification.from_pretrained(self.model_type, force_download=True)
        return model

    def load_transformer(self):
        self._transformer = self._get_transformer()

    



class FactCCBertBaseUncased(FactCCBase):
    def __init__(self, config_path):
        super().__init__(config_path)
        
        self._tokenizer = None
        self._transformer = None

    @property
    def model_type(self):
        # return self._model_type
        return "bert-base-uncased"

    @property
    def tokenizer(self):
        return self._tokenizer
    
    def _get_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_type)
        return tokenizer

    @property
    def transformer(self):
        return self._transformer
    
    def _get_transformer(self):
        model = TFBertForSequenceClassification.from_pretrained(self.model_type, force_download=True)
        return model

    def load_transformer(self):
        self._transformer = self._get_transformer()

    def load_tokenizer(self):
        self._tokenizer =  self._get_tokenizer()


class FactCCBertBaseFineTuned(FactCCBase):
    def __init__(self, config_path):
        super().__init__(config_path)
        
        self._tokenizer = None
        self._transformer = None

    @property
    def model_type(self):
        # return self._model_type
        return "bert-base-cased"

    @property
    def tokenizer(self):
        return self._tokenizer
    
    def _get_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_type)
        return tokenizer

    @property
    def transformer(self):
        return self._transformer
    
    
    def _get_transformer(self):
        model = TFBertForSequenceClassification.from_pretrained(self.model_type, force_download=True)
        return model
    
    def train_model(self):
        self._model = self.transformer
        for w in self._model.bert.weights:
            w._trainable= False
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self._model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    def load_transformer(self):
        self._transformer = self._get_transformer()

    def load_tokenizer(self):
        self._tokenizer =  self._get_tokenizer()



class FactCCRoBERTa(FactCCBase):
    def __init__(self, config_path):
        super().__init__(config_path)
        
        self._tokenizer = None
        self._transformer = None

    @property
    def model_type(self):
        # return self._model_type
        return 'roberta-base'

    @property
    def tokenizer(self):
        return self._tokenizer
    
    def _get_tokenizer(self):
        tokenizer = RobertaTokenizer.from_pretrained(self.model_type)
        return tokenizer

    @property
    def transformer(self):
        return self._transformer
    
    
    def _get_transformer(self):
        model = TFRobertaForSequenceClassification.from_pretrained(self.model_type, force_download=True)
        return model
    
    def load_model_from_file(self):
        self._model = RobertaForSequenceClassification.from_pretrained(self.output_dir, from_tf=True)

    def load_transformer(self):
        self._transformer = self._get_transformer()

    def load_tokenizer(self):
        self._tokenizer =  self._get_tokenizer()



class FactCCDistilBert(FactCCBase):
    def __init__(self, config_path):
        super().__init__(config_path)
        
        self._tokenizer = None
        self._transformer = None

    @property
    def model_type(self):
        # return self._model_type
        return 'distilbert-base-uncased'

    @property
    def tokenizer(self):
        return self._tokenizer
    
    def _get_tokenizer(self):
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_type)
        return tokenizer

    @property
    def transformer(self):
        return self._transformer
    
    def load_model_from_file(self):
        self._model = DistilBertForSequenceClassification.from_pretrained(self.output_dir, from_tf=True)

    def _get_transformer(self):
        model = TFDistilBertForSequenceClassification.from_pretrained(self.model_type, force_download=True)
        return model

    def load_transformer(self):
        self._transformer = self._get_transformer()

    def load_tokenizer(self):
        self._tokenizer =  self._get_tokenizer()



# class FactCCRoBerta(FactCCBase):
