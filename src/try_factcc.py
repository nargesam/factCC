import click
# import logging
import pathlib 
import os
import torch

import diskcache as dc

from transformers import BertForSequenceClassification, BertTokenizer
from prefect import task, Task, Flow, Parameter
from prefect.engine.signals import SKIP

from model import FactCCBertBaseCased, FactCCBertBaseUncased
from visualization import FactCCViz

from colorama import Fore, Style


def colored(string,color):
    return color + string+Fore.RESET

# def get_model(model_type, config_path):
#     model_map = {
#         'bert-base-cased': FactCCBertBaseCased,
#         'bert-base-uncased': FactCCBertBaseUncased
#     }

#     return model_map[model_type](config_path=config_path)


@click.command()
# @click.option('-n', '--workflow-name', type=str, required=True)
# @click.option('-m', '--model-type', type=str, required=True)
# @click.option('-c', '--config-path', type=str, required=False)
# def run(workflow_name, config_path, model_type):
    # obj = FactCCBertBaseCased(cfgpath)

def run_test():
    click.echo('Welcome to Orient! Let\'s get you some recommendations.')
    name = click.prompt(colored('Enter your name',Fore.MAGENTA))

    # obj = get_model(model_type=model_type, config_path=config_path)



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


