import click
import pathlib 
import os
import random
import re
import textwrap

import diskcache as dc

from colorama import Fore, Style
from transformers import BertForSequenceClassification, BertTokenizer
from prefect import task, Task, Flow, Parameter
from prefect.engine.signals import SKIP

import pandas as pd
import torch

from model import FactCCBertBaseCased, FactCCBertBaseUncased, FactCCRoBERTa

pd.set_option('display.max_columns', 1000)  # or 1000
pd.set_option('display.max_rows', 1000)  # or 1000
pd.set_option('display.max_colwidth', 199)  # or 199

def colored(string, color):
    return color + string + Fore.RESET


def get_model(model_type, config_path):
    model_map = {
        'bert-base-cased': FactCCBertBaseCased,
        'bert-base-uncased': FactCCBertBaseUncased,
        'roberta-base': FactCCRoBERTa
    }
    return model_map[model_type](config_path=config_path)


def pred_label(pred):
    if pred:
        return "SUPPORTS"
    else:
        "REFUTES"


@click.command()
@click.option('-m', '--model-type', type=str, required=True)
@click.option('-c', '--config-path', type=str, required=False)

def run_test(config_path, model_type):
    
    obj = get_model(model_type=model_type, config_path=config_path)
    obj.load_tokenizer()
    obj.load_model_from_file()
    
    while True:
        saved_test = click.confirm('\n \n Do you want to use our test data?')

        if saved_test:
            obj.load_test_data()
            rownum = random.randint(0, len(obj._test_data))
            # print(rownum)
            s = obj._test_data.iloc[rownum]

            extraction = str(s['extraction_span'])
            extraction = re.split('\W+', extraction)
            low, high = extraction[1], extraction[2] 
            
            click.echo(colored(" \n Here's a randomly selected  news: \n", Fore.MAGENTA))
            news = s['sentence1']
            news_split = news.split()
            
            before = ' '.join(news_split[: int(low)])
            bold = ' '.join(news_split[int(low):int(high)])
            after = ' '.join(news_split[int(high):])
            
            to_echo = before + '\033[1m' + bold + '\033[0m' + after
            
            click.echo(to_echo+ '\n ')
            # click.echo(colored(to_echo, Fore.MAGENTA))

            click.echo(" \n and its claim: \n")
            click.echo(colored(s['sentence2'], Fore.MAGENTA))

            pred = obj.run_test_cases(str(s['sentence1']), str(s['sentence1']))
            result =  '\n \n ' + str(s['label']) + " was the actual result and FactCC predicted: " + pred_label(pred) + '\n\n'
            
            click.echo(colored(result, Fore.WHITE))
        
        else: 
            sentence1 = click.prompt("Please enter your long text")
            sentence2 = click.prompt("Please enter your claim text")
            label = click.prompt("SUPPORTS/REFUTES?")

            pred = obj.run_test_cases(sentence1, sentence2)

            result = str(label) + "was the actual result and FactCC predicted:" + pred_label(pred)
            click.echo(colored(result, Fore.WHITE))

        if_again = click.confirm(" \n \n Would you like to try again? ")

        if if_again:
            continue
        else:
            exit()


if __name__ == '__main__':
    # Must add this environment variable or everything explodes
    # HT: https://github.com/dmlc/xgboost/issues/1715
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    click.echo(colored('\n Welcome to FactCC!', Fore.MAGENTA))
    
    run_test()