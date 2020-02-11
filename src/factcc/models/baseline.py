import configparser
import datetime
import json
import os
import pathlib
import pickle
import sys

import numpy
import pandas as pd
import tensorflow as tf

import torch 

from transformers import *

# Must add this environment variable or everything explodes
# HT: https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[3]
# /Users/ns5kn/Documents/insight/projects/factCC/src/models/bertviz/bertviz/head_view.py
# bertvizpath = os.path.join(PROJECT_DIR, 'src/models/bertviz/' )
# sys.path.insert(0, bertvizpath)

# from bertviz import head_view

# current_time = datetime.datetime.now()

# cfgpath = os.path.join(PROJECT_DIR, './configuration.cfg' )
# config = configparser.RawConfigParser()
# try:
#     config.read(cfgpath)
# except:
#     print("Couldn't read config file from ./configuration.cfg")
#     exit()

# tokenizer_mode = config.get('Params', 'tokenizer_mode')
# dataset_File = config.get('Params', 'dataset')
# csvfile = config.get('Params', 'csvfile')
# classification_mode = config.get('Params', 'classification_mode')
# maxlen = int(config.get('Params', 'maxlen'))
# batchsize = int(config.get('Params', 'batchsize'))
# steps = int(config.get('Params', 'steps'))
# num_epochs = int(config.get('Params', 'num_epochs'))



def read_jsonl(datafolder):
    with open(datafolder) as f:
        dataset = [json.loads(line) for line in f]
    df = pd.DataFrame(dataset)
    df = df.rename(columns={'text': 'sentence1', 'claim': 'sentence2', 'id': 'idx'})
    df = df[['idx', 'sentence1', 'sentence2', 'label']]
    return df

def read_csv(csvfolder):
    data = pd.read_csv(csvfolder)
    data =  data.rename(columns={'id': 'idx', 'text': 'sentence1', 'claim': 'sentence2'})
    data = data[['idx', 'sentence1', 'sentence2', 'label']]
    return data
    

def train_test(data):
    l = len(data)
    to = int(0.8*(l))
    train = data[:to]
    validation = data[to:]
    train["label"] = train["label"].map(
                    {
                        'CORRECT': numpy.int64(1),
                        'INCORRECT': numpy.int64(0),
                        'SUPPORTS': numpy.int64(1),
                        'REFUTES': numpy.int64(0)
                    },
                )
    validation["label"] = validation["label"].map(
        {
            'CORRECT': numpy.int64(1),
            'INCORRECT': numpy.int64(0),
            'SUPPORTS': numpy.int64(1),
            'REFUTES': numpy.int64(0)
        },
    )

    print(train.head(1))
    print(train['label'].value_counts())
    return train, validation

def prepare_for_model(dataset):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_mode)
    tfdata = tf.data.Dataset.from_tensor_slices(dict(dataset))
    model_dataset = glue_convert_examples_to_features(tfdata, tokenizer, label_list=[numpy.int64(1), numpy.int64(0)], max_length=maxlen , task='mrpc', output_mode="classification")
    return model_dataset


def create_model():
    model = TFBertForSequenceClassification.from_pretrained(classification_mode, force_download=True)

    # for param in model.bert.parameters():
    #     param.requires_grad = False

    # for w in model.bert.weights():
    #     w._trainable= False
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model


def train_model(model, train_dataset, valid_dataset):
    train_dataset = train_dataset.shuffle(100).batch(batchsize).repeat(2)
    valid_dataset = valid_dataset.batch(batchsize).repeat()

    # creating callback
    folder_name = str(current_time.day) + "-" + str(current_time.hour) + "-" + str(current_time.minute)
    # print(folder_name)
    respath = os.path.join(PROJECT_DIR, 'models/saved_models', folder_name)
    # print(respath)
    # best_path = respath + 'crf.hdf5'
    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(best_path, monitor='val_acc', verbose=1,
    #                      save_best_only=True, mode='max', period=1)
    # callbacks=[model_checkpoint]
    history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=steps,
                    validation_data=valid_dataset, validation_steps=2)

    

    if not os.path.exists(respath):
        os.makedirs(respath) 
    
    histpath = os.path.join(respath, 'history.pickle')
    # print(histpath)
    with open(histpath, 'wb') as f:
            pickle.dump(history.history, f)

    # Load the TensorFlow model in PyTorch for inspection
    # modelpath = os.path.join(PROJECT_DIR, 'models/saved_models/test/')
    model.save_pretrained(respath)

    return respath


def test_data(modelpath, sent1, sent2):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    pytorch_model = BertForSequenceClassification.from_pretrained(modelpath, from_tf=True)

    inputs_1 = tokenizer.encode_plus(sent1, sent2, add_special_tokens=True, return_tensors='pt')
    pred_1 = pytorch_model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
    # pred_2 = pytorch_model(inputs_2['input_ids'], token_type_ids=inputs_2['token_type_ids'])[0].argmax().item()

    print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")

    return pytorch_model, pred_1

def show_head_view(model, sentence_a, sentence_b=None):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_mode)

    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    # (tensor([[-0.1247,  0.1027]], grad_fn=<AddmmBackward>),)
    attention = model(input_ids, token_type_ids=token_type_ids)[0]
    attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    print(attention)
    # exit()
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    if sentence_b:
        sentence_b_start = token_type_ids[0].tolist().index(1)
    else:
        sentence_b_start = None
    head_view(attention, tokens, sentence_b_start)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()





# data = read_csv(csvfolder)
# print("0- reading Jsonl data - ")

# data = read_jsonl(datafolder)
# print("1- Jsonl data - done")
# # print(data.head(2))
# # exit()
# train, validation = train_test(data)
# # print(len(train))
# # print(data.head(2))
# # exit()
# train = train[:10]
# validation = validation[10:12]
# train_dataset = prepare_for_model(train)
# print("2- Train to tf.Dataset and tokenization Done - ")

# validation_dataset = prepare_for_model(validation)
# print("3- Validation to tf.Dataset and tokenization Done - ")

# model = create_model()
# print(model.summary())
# print("4- Now Training the mode ")

# path = train_model(model, train_dataset, validation_dataset )
# print("5- Training Done ")

# print(path)

if __name__ == "__main__":
    # datafolder = os.path.join(PROJECT_DIR, dataset_File )
    # csvfolder  = os.path.join(PROJECT_DIR , csvfile)
    # print("0- reading Jsonl data - ")
    
    # data = read_csv(csvfolder)

    # # data = read_jsonl(datafolder)
    # print("1- Jsonl data - done")
    # # # print(data.head(2))
    # # # exit()
    # train, validation = train_test(data)
    # print(len(train))
    # # # print(data.head(2))
    # # # exit()
    # train = train[:10]
    # validation = validation[10:12]
    # train_dataset = prepare_for_model(train)
    # print("2- Train to tf.Dataset and tokenization Done - ")

    # print(type(train_dataset))
    # # exit()
    # validation_dataset = prepare_for_model(validation)
    # print("3- Validation to tf.Dataset and tokenization Done - ")

    # model = create_model()
    # print(model.summary())
    # print("4- Now Training the mode ")

    # path = train_model(model, train_dataset, validation_dataset )
    # print("5- Training Done ")
    # print(path)
    





    modelpath = '/Users/ns5kn/Documents/insight/projects/factcc/models/saved_models/29-csvfile_5batch_7epoch/'
    sentence_0 = "This research was consistent with his findings."
    sentence_1 = "His findings were compatible with this research."
    sentence_0 = "(CNN) Georgia Southern University was in mourning Thursday after five nursing students were killed the day before in a multivehicle wreck near Savannah. Caitlyn Baggett, Morgan Bass, Emily Clark, Abbie Deloach and Catherine (McKay) Pittman -- all juniors -- were killed in the  Wednesday morning crash as they were traveling to a hospital in Savannah, according to the school website. "
    sentence_1 = "georgia southern university was in mourning after five nursing students died."
    model, pred = test_data(modelpath, sentence_0, sentence_1)
    # print(model)
    print(pred)
    # show_head_view(model, sentence_0, sentence_1)


# # print(train.head(2))
# # exit()

# # features = ["idx", "text", "claim","label"]

# train_data = tf.data.Dataset.from_tensor_slices(dict(train))
# validation_data = tf.data.Dataset.from_tensor_slices(dict(validation))

# print("Prepare dataset for GLUE")

# train_dataset = glue_convert_examples_to_features(train_data, tokenizer, label_list=[numpy.int64(1), numpy.int64(0)], max_length=maxlen , task='mrpc', output_mode="classification")
# print("Done Glue for trian")
# valid_dataset = glue_convert_examples_to_features(validation_data, tokenizer, label_list=[numpy.int64(1), numpy.int64(0)] , max_length=maxlen, task='mrpc')
# print("Done Glue for validation")

# train_dataset = train_dataset.shuffle(100).batch(64).repeat(2)
# valid_dataset = valid_dataset.batch(64)
# print("Prepare dataset for GLUE: DONE")


# # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
# print("Optimizer")
# model = TFBertForSequenceClassification.from_pretrained(classification_mode, force_download=True)
# optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
# print(model.summary())

# Train and evaluate using tf.keras.Model.fit()
# history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=steps,
#                     validation_data=valid_dataset, validation_steps=2)

# respath = os.path.join(PROJECT_DIR, 'models/saved_models/test/history.pickle')
# with open(respath, 'wb') as f:
#         pickle.dump(history.history, f)

# # Load the TensorFlow model in PyTorch for inspection
# modelpath = os.path.join(PROJECT_DIR, 'models/saved_models/test/')
# model.save_pretrained(modelpath)


# Load and test the saved model
# pytorch_model = BertForSequenceClassification.from_pretrained('/Users/ns5kn/Documents/insight/projects/factcc/models/', from_tf=True)
# print("model done")


# # Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
# sentence_0 = "This research was consistent with his findings."
# sentence_1 = "His findings were compatible with this research."
# sentence_2 = "His findings were not compatible with this research."
# inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
# inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')


# print(inputs_1)


# pred_1 = pytorch_model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
# # print(pred_1)

# # 
# pred_2 = pytorch_model(inputs_2['input_ids'], token_type_ids=inputs_2['token_type_ids'])[0].argmax().item()

# print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
# print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")


