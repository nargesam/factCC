# AutoFact:  Automated tool for evaluating the factual consistency of summarized text

Project for the AI Fellowship at Insight

In this project I introduce AutoFact which is an automated tool for evaluating the factual consistency of summarizad text. 

This project is built based on this paper from Salesforce Research: https://arxiv.org/abs/1910.12840 

<!-- ===================================================== -->
## Project description

AutoFact: a tool that evaluates the factual consistency between the summary of a large text document and its source using Transformer models.

Development: used HuggingFace to built the AutoFact model using various Transformer model and compare various Transormer models such as Google BERT, Facebook RoBERTa and DistilBert to demonsterate runtime-accuracy trade off.

Demo: Served via an interactive command line interface created with the Python package Click.

<!-- =================================================== -->

<img src="https://github.com/nargesam/factCC/blob/master/images/model.jpg" width="300" height="300">

<!-- ===================================================== -->
## Data description
The source text is CNN and Daily Mail news. The claim is an abstractive summarization of those news. After the summarization has been created and were labeled "SUPPORTS", they have all been augmented to create false claims and were labeled "REFUTES". Here's a description of data: 

<img src="https://github.com/nargesam/factCC/blob/master/images/data.png" width="300" height="300">


The data is provided from Salesforce research, and you can re-create the dataset here: https://github.com/salesforce/factCC/tree/master/data_generation

<!-- ==================================================== -->

## Test AutoFact

To try AutoFact, after you cloned the repo: 

```
git clone https://github.com/nargesam/factCC.git
```

You can recreate the conda environment:

```
conda env create -f factcc_environment.yml 
```

Or, install the requirement.txt file:

```
pip3 install -r requirements.txt
```

Please download the [BERT Base Cased](https://insight-ai-factcc.s3-us-west-2.amazonaws.com/factcc/models/saved_models/4-18-7-10perc-bert-cased-6epoch-8step-50batch-73acc/tf_model.h5) model and its [config file](https://insight-ai-factcc.s3-us-west-2.amazonaws.com/factcc/models/saved_models/4-18-7-10perc-bert-cased-6epoch-8step-50batch-73acc/config.json), [BERT Base Uncased](https://insight-ai-factcc.s3-us-west-2.amazonaws.com/factcc/models/saved_models/04-22-07-Bert-Base-UnCased-10perc-6epoch-8step-50batch-72acc/tf_model.h5) model and its [config file](https://insight-ai-factcc.s3-us-west-2.amazonaws.com/factcc/models/saved_models/04-22-07-Bert-Base-UnCased-10perc-6epoch-8step-50batch-72acc/config.json), or [RoBERTa](https://insight-ai-factcc.s3-us-west-2.amazonaws.com/factcc/models/saved_models/5-18-37-Roberta-10perc-6epoch-8step-50batch-68acc/tf_model.h5) model and its [config file](https://insight-ai-factcc.s3-us-west-2.amazonaws.com/factcc/models/saved_models/5-18-37-Roberta-10perc-6epoch-8step-50batch-68acc/config.json) and save them to their directories: models/saved_models/< model-type >/batch_size-50__epoch-6__datasize-10perc

< model-type >: bert-base-cased, bert-base-uncased, roberta-base
< Config Path >: src/factcc/try_factcc.cfg

Run the test python file: 

```
python src/factcc/try_factcc.py  --model-type < model-type > --config-path < Config Path >

```
