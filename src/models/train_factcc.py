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
# from transformers import BertForSequenceClassification
# import tensorflow_datasets


os.environ['KMP_DUPLICATE_LIB_OK']='True'
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
current_time = datetime.datetime.now()

cfgpath = os.path.join(PROJECT_DIR, './configuration.cfg' )
config = configparser.RawConfigParser()
try:
    config.read(cfgpath)
except:
    print("Couldn't read config file from ./configuration.cfg")
    exit()

tokenizer_mode = config.get('Params', 'tokenizer_mode')
dataset_File = config.get('Params', 'dataset')
csvfile = config.get('Params', 'csvfile')
classification_mode = config.get('Params', 'classification_mode')
maxlen = int(config.get('Params', 'maxlen'))
batchsize = int(config.get('Params', 'batchsize'))
steps = int(config.get('Params', 'steps'))
num_epochs = int(config.get('Params', 'num_epochs'))



def read_jsonl(datafolder):
    with open(datafolder) as f:
        dataset = [json.loads(line) for line in f]
    df = pd.DataFrame(dataset)
    return df

def read_csv(csvfolder):
    data = pd.read_csv(csvfolder)
    data = data[['idx', 'sentence1', 'sentence2', 'label']]
    return data
    


def train_test(data):
    train = data[:10]
    validation = data[10:12]
    train["label"] = train["label"].map({'SUPPORTS': numpy.int64(1) ,'REFUTES': numpy.int64(0)})
    validation["label"] = validation["label"].map({'SUPPORTS': numpy.int64(1) ,'REFUTES': numpy.int64(0)})
    return train, validation

def prepare_for_model(dataset):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_mode)
    tfdata = tf.data.Dataset.from_tensor_slices(dict(dataset))
    model_dataset = glue_convert_examples_to_features(tfdata, tokenizer, label_list=[numpy.int64(1), numpy.int64(0)], max_length=maxlen , task='mrpc', output_mode="classification")
    return model_dataset

def create_model():
    model = TFBertForSequenceClassification.from_pretrained(classification_mode, force_download=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model


def train_model(model, train_dataset, valid_dataset):
    train_dataset = train_dataset.shuffle(100).batch(64).repeat(2)
    valid_dataset = valid_dataset.batch(64)
    history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=steps,
                    validation_data=valid_dataset, validation_steps=2)

    folder_name = str(current_time.day) + "-" + str(current_time.hour) + "-" + str(current_time.minute)
    respath = os.path.join(PROJECT_DIR, 'models/saved_models', folder_name, 'history.pickle')
    with open(respath, 'wb') as f:
            pickle.dump(history.history, f)

    # Load the TensorFlow model in PyTorch for inspection
    # modelpath = os.path.join(PROJECT_DIR, 'models/saved_models/test/')
    model.save_pretrained(respath)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()



datafolder = os.path.join(PROJECT_DIR, dataset_File )
csvfolder  = os.path.join(PROJECT_DIR , csvfile)


data = read_csv(csvfolder)
train, validation = train_test(data)
train_dataset = prepare_for_model(train)
validation_dataset = prepare_for_model(validation)
model = create_model()
print(model.summary())
train_model(model, train_dataset, validation_dataset )




# import datetime  
    
# # using now() to get current time  
# current_time = datetime.datetime.now()  
    
# # Printing attributes of now().  
# print ("The attributes of now() are : ")  
    
# print ("Year : ", end = "")  
# print (current_time.year)  
    
# print ("Month : ", end = "")  
# print (current_time.month)  
    
# print ("Day : ", end = "")  
# print (current_time.day)  
    
# print ("Hour : ", end = "")  
# print (current_time.hour)  
    
# print ("Minute : ", end = "")  
# print (current_time.minute)  
    
# print ("Second : ", end = "")  
# print (current_time.second)  
    
# print ("Microsecond : ", end = "")  
# print (current_time.microsecond)  


# print('after')

# data = tensorflow_datasets.load('glue/mrpc', shuffle_files=True)
# print('checkpint on Pandas')
# # filepath = os.path.join(PROJECT_DIR, 'data/interim/sample_train.csv')

# print('read the data: DONE')




# # 'SUPPORTS' == 1, 'REFUTES' == 0



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


