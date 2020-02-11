import json
import os

import pandas as pd

import util

def load_dataset(data_file=None):
    if data_file is None:
        dl = util.DirectoryLookup()
        data_file = os.path.join(dl.processed_data_dir, 'paired_data/data-clipped/data-dev.jsonl')

    with open(data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]
    # df = pd.read_csv(data_file) 
    df = pd.DataFrame(dataset)
    df['augmentation'] = df['augmentation'].fillna('None')

    return df