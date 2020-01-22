
import pandas as pd
from transformers import *

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

        guid = row[0]
        text_a = row[1]
        text_b = row[2]
        if labels_available:
            labels = list(row[3:])
            print(type(labels))
            print(labels)

            label_map = {label: i for i, label in enumerate(labels)}
            print(label_map)
            break
        else:
            labels = [0]*num_labels
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b , label=labels))
    return examples

train_d = create_examples(train)    