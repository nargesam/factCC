import os
import pathlib

import numpy
import pandas as pd
import tensorflow as tf
import torch 

from transformers import *
# from transformers import BertForSequenceClassification
# import tensorflow_datasets


os.environ['KMP_DUPLICATE_LIB_OK']='True'
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]


# from tokenizers import BertWordPieceTokenizer
# import tokenizers

# Load dataset, tokenizer, model from pretrained model/vocabulary
# print('before')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print('after')

# data = tensorflow_datasets.load('glue/mrpc', shuffle_files=True)
print('checkpint on Pandas')
filepath = os.path.join(PROJECT_DIR, 'data/interim/sample_train.csv')
data = pd.read_csv(filepath)
print('read the data')
data = data[['idx', 'sentence1', 'sentence2', 'label']]
print('read the data: DONE')

train = data[:100000]
validation = data[100000:]


# # 'SUPPORTS' == 1, 'REFUTES' == 0
train["label"] = train["label"].map({'SUPPORTS': numpy.int64(1) ,'REFUTES': numpy.int64(0)})
validation["label"] = validation["label"].map({'SUPPORTS': numpy.int64(1) ,'REFUTES': numpy.int64(0)})


# # print(train.head(2))
# # exit()

# # features = ["idx", "text", "claim","label"]

train_data = tf.data.Dataset.from_tensor_slices(dict(train))
validation_data = tf.data.Dataset.from_tensor_slices(dict(validation))

print("Prepare dataset for GLUE")

train_dataset = glue_convert_examples_to_features(train_data, tokenizer, label_list=[numpy.int64(1), numpy.int64(0)], max_length=512 , task='mrpc', output_mode="classification")
print("Done Glue for trian")
valid_dataset = glue_convert_examples_to_features(validation_data, tokenizer, label_list=[numpy.int64(1), numpy.int64(0)] , max_length=512, task='mrpc')
print("Done Glue for validation")

train_dataset = train_dataset.shuffle(100).batch(64).repeat(2)
valid_dataset = valid_dataset.batch(64)
print("Prepare dataset for GLUE: DONE")


# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
print("Optimizer")
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', force_download=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
print(model.summary())

# Train and evaluate using tf.keras.Model.fit()
history = model.fit(train_dataset, epochs=8, steps_per_epoch=12,
                    validation_data=valid_dataset, validation_steps=2)


# Load the TensorFlow model in PyTorch for inspection
modelpath = os.path.join(PROJECT_DIR, 'models/saved_models/')
model.save_pretrained(modelpath)


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


