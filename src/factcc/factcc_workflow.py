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

# PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
# cfgpath = os.path.join(PROJECT_DIR, './configuration.cfg' )
# from dotenv import find_dotenv, load_dotenv


@task
def test_pred_existance_check(obj):
    if obj.test_pred_exists():
        raise SKIP("Test data has already been created.")

@task
def model_existence_check(obj):
    if obj.model_exists():
        raise SKIP("Model has already been created.")

@task
def load_data(obj):
    obj.load_data(clobber=False)

@task
def split_train_test(obj):
    obj.split_train_test(clobber=False)
    # obj.train_dataset, obj.validation_dataset

@task(skip_on_upstream_skip=False)
def load_tokenizer(obj):
    obj.load_tokenizer()

@task
def convert_examples_to_features(obj):
    obj.convert_examples_to_features(clobber=False)

@task
def load_transformer(obj):
    obj.load_transformer()

@task
def create_model(obj):
    obj.create_model()

@task
def train_model(obj):
    obj.train_model()

@task
def save_model(obj):
    obj.save_model()


@task
def load_model_from_file(obj):
    obj.load_model_from_file()

@task
def load_test_data(obj):
    obj.load_test_data()


@task
def run_test(obj):
    obj.run_test()


@task
def save_test(obj):
    obj.save_test()


def get_model(model_type, config_path):
    model_map = {
        'bert-base-cased': FactCCBertBaseCased,
        'bert-base-uncased': FactCCBertBaseUncased
    }

    return model_map[model_type](config_path=config_path)


@click.command()
@click.option('-n', '--workflow-name', type=str, required=True)
@click.option('-m', '--model-type', type=str, required=True)
@click.option('-c', '--config-path', type=str, required=False)
def run(workflow_name, config_path, model_type):
    # obj = FactCCBertBaseCased(cfgpath)

    obj = get_model(model_type=model_type, config_path=config_path)

    with Flow(workflow_name)  as f:
        # Train Workflow
        task_test_pred_existence_check = test_pred_existance_check(obj=obj)

        task_model_existence_check = model_existence_check(obj=obj) #, upstream_tasks=[task_test_pred_existence_check])

        task_load_data = load_data(obj=obj, upstream_tasks=[task_model_existence_check])

        task_load_tokenizer = load_tokenizer(obj=obj, upstream_tasks=[task_model_existence_check]) #, upstream_tasks=[task_test_pred_existence_check])
        task_load_transformer = load_transformer(obj=obj, upstream_tasks=[task_model_existence_check])

        task_split_train_validation = split_train_test(obj=obj, upstream_tasks=[task_load_data])

        task_convert_examples_to_features = convert_examples_to_features(
            obj=obj, 
            upstream_tasks=[task_split_train_validation, task_load_tokenizer]
        )

        task_create_model = create_model(obj=obj, upstream_tasks=[task_convert_examples_to_features, task_load_transformer])

        task_train_model = train_model(obj=obj, upstream_tasks=[task_create_model])
        
        task_save_model = save_model(obj=obj, upstream_tasks=[task_train_model])

        # Test workflow: load model from file
        task_load_model_from_file = load_model_from_file(obj=obj) #, upstream_tasks=[task_save_model] )#, upstream_tasks=[task_test_pred_existence_check])
        task_load_test_data = load_test_data(obj=obj) #, upstream_tasks=[task_save_model])

        task_run_test = run_test(obj=obj, upstream_tasks=[task_load_model_from_file, task_load_test_data])

        task_save_test = save_test(obj=obj, upstream_tasks=[task_run_test])

    flow_state = f.run()

    
    # shell_output = flow_state.result[task_run_test].result
    # print(shell_output)

if __name__ == '__main__':
    # Must add this environment variable or everything explodes
    # HT: https://github.com/dmlc/xgboost/issues/1715
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    run()


