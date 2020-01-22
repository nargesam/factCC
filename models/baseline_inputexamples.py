import tensorflow as tf
import tensorflow_datasets
from transformers import *
import pandas as pd
# from tokenizers import BertWordPieceTokenizer
# import tokenizers

"""
tensorflow.python.data.ops.dataset_ops._OptionsDataset is just another class extending the base class tf.compat.v2.data.Dataset (DatasetV2) which holds tf.data.Options along with the original tf.compat.v2.data.Dataset dataset (The Portuguese-English tuples in your case).

(tf.data.Options operates when you are using streaming functions over your dataset tf.data.Dataset.map or tf.data.Dataset.interleave)

How to view the individual elements?

I'm sure there are many ways, but one straight way would be to use the iterator in the base class:

Since examples['train'] is a type of _OptionsDataset here is iterating by calling a method from tf.compat.v2.data.Dataset

iterator = examples['train'].__iter__()
next_element = iterator.get_next()
pt = next_element[0]
en = next_element[1]
print(pt.numpy())
print(en.numpy())
"""

# Load dataset, tokenizer, model from pretrained model/vocabulary
print('before')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', force_download=True)
print('after')

# data = tensorflow_datasets.load('glue/mrpc', shuffle_files=True)


print('checkpint on Pandas')
data = pd.read_csv('/Users/ns5kn/Documents/insight/projects/factcc/data/interim/sample_train.csv')
print('read the data')
print(data.columns)
# ['id', 'claim', 'label', 'extraction_span', 'backtranslation', 'augmentation', 'augmentation_span', 'noisy', 'filepath', 'text']
data = data[['idx', 'text', 'claim', 'label']]
# data = data.rename({'id': 'idx'})
print('read the data')

train = data[:1000000]
validation = data[1000000:]



# def __init__(self, guid, text_a, text_b, labels=None):
#       """Constructs a InputExample.

#       Args:
#       guid: Unique id for the example.
#       text_a: string. The untokenized text of the first sequence. For single
#       sequence tasks, only this sequence must be specified.
#       text_b: (Optional) string. The untokenized text of the second sequence.
#       Only must be specified for sequence pair tasks.
#       labels: (Optional) [string]. The label of the example. This should be
#       specified for train and dev examples, but not for test examples.
#       """
#       self.guid = guid
#       self.text_a = text_a
#       self.text_b = text_b
#       self.labels = labels

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output



# print(train.head(2))
def create_examples(df, labels_available=True, num_labels=2):
    """Creates examples for the training and dev sets."""
    examples = []
    for index, row in df.iterrows():
        # print(row)
        # print(row[0])
        # break

        guid = str(row[0])
        text_a = str(row[1])
        text_b = str(row[2])
        if labels_available:
            label = str(row[3])
            # label_map = {label: i for i, label in enumerate(labels)}
            # print(label_map)
            # break
        else:
            label = '0'
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b , label=label))
    return examples
print('checkpint on creat_examples')
train_data = create_examples(train)
validation_data = create_examples(validation)
print('checkpint on creat_examples:DONE')

# Prepare dataset for GLUE as a tf.data.Dataset instance
# print(type(data['train']))
# train_data = data['train']
print("Prepare dataset for GLUE")

train_dataset = glue_convert_examples_to_features(train_data, tokenizer, label_list=('SUPPORTS', 'REFUTES'), max_length=128 , task='mrpc')
valid_dataset = glue_convert_examples_to_features(validation_data, tokenizer, label_list=('SUPPORTS', 'REFUTES') , max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
valid_dataset = valid_dataset.batch(64)
print("Prepare dataset for GLUE: DONE")


# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
print("Optimizer")

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
print("loss")

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
print("Prepare training")



# Train and evaluate using tf.keras.Model.fit()
history = model.fit(train_dataset, epochs=2, steps_per_epoch=10,
                    validation_data=valid_dataset, validation_steps=2)


print("Prepare training")

# Load the TensorFlow model in PyTorch for inspection
model.save_pretrained('/save/')
pytorch_model = BertForSequenceClassification.from_pretrained('./save/', from_tf=True)

print("model done")



# Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
sentence_0 = "This research was consistent with his findings."
sentence_1 = "His findings were compatible with this research."
sentence_2 = "His findings were not compatible with this research."
inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')

pred_1 = pytorch_model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
pred_2 = pytorch_model(inputs_2['input_ids'], token_type_ids=inputs_2['token_type_ids'])[0].argmax().item()

print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")


