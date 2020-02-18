import click
import pathlib 
import os
import re
import torch

import diskcache as dc

from colorama import Fore, Style
from transformers import BertForSequenceClassification, BertTokenizer
from prefect import task, Task, Flow, Parameter
from prefect.engine.signals import SKIP

from model import FactCCBertBaseCased, FactCCBertBaseUncased
# from visualization import FactCCViz



def colored(string,color):
    return color + string + Fore.RESET


def get_model(model_type, config_path):
    model_map = {
        'bert-base-cased': FactCCBertBaseCased,
        'bert-base-uncased': FactCCBertBaseUncased
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
    click.echo(colored('\n Welcome to FactCC!', Fore.MAGENTA))
    
    obj = get_model(model_type=model_type, config_path=config_path)
    obj.load_tokenizer()
    obj.load_model_from_file()
    saved_test = click.confirm('Do you want to use our test data?')

    if saved_test:

        obj.load_test_data()
        sample = obj._test_data.sample()

        # extraction = str(sample['extraction_span'])
        # extraction = re.split('\W+', extraction)
        # low, high = extraction[1], extraction[2]

        # TODO: better show the sentence 1
        # print(sample['sentence1'])
        # sample_news = str(sample['sentence1']).split()

        click.echo(" Here's a randomly selected  news: \n \n")
        click.echo(colored(sample['sentence1'], Fore.MAGENTA))

        click.echo(" and its claim: \n \n")
        click.echo(colored(sample['sentence2'], Fore.MAGENTA))

        pred = obj.run_test_cases(str(sample['sentence1']), str(sample['sentence1']))

        result = str(sample['label']) + "was the actual result and FactCC predicted:" + pred_label(pred)
        
        click.echo(colored(result, Fore.WHITE))
    
    else: 

        sentence1 = click.prompt("Please enter your long text")
        sentence2 = click.prompt("Please enter your claim text")
        label = click.prompt("SUPPORTS/REFUTES?")

        pred = obj.run_test_cases(sentence1, sentence2)

        result = str(label) + "was the actual result and FactCC predicted:" + pred_label(pred)
        click.echo(colored(result, Fore.WHITE))







        # Test workflow: load model from file
# task_load_model_from_file = load_model_from_file(obj=obj) #, upstream_tasks=[task_save_model] )#, upstream_tasks=[task_test_pred_existence_check])
# task_load_test_data = load_test_data(obj=obj) #, upstream_tasks=[task_save_model])

# task_run_test = run_test(obj=obj, upstream_tasks=[task_load_model_from_file, task_load_test_data])

# task_save_test = save_test(obj=obj, upstream_tasks=[task_run_test])


if __name__ == '__main__':
    # Must add this environment variable or everything explodes
    # HT: https://github.com/dmlc/xgboost/issues/1715
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    run_test()


    # run()


