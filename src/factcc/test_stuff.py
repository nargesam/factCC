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


def load_model_from_file(self):
        model = BertForSequenceClassification.from_pretrained(, from_tf=True)
