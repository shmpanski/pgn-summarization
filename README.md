# Pytorch Pointer-Generator Recurrent NN
## Model
It's an implementation of [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368) model.
## Requirements
* python3
* pytorch4
* tensorboardX

## Usage
### Preparation
For training it's necessary to prepare `train.tsv` and `validation.tsv` parts of dataset and store in common folder. Each line of the document has two parts: *source* and  *target* separated by `tab` symbol. Tabular files don't need headers.
### Training
For train model use
```
$ python main.py
```
#### Parameters
Full list of parameters can be obtained using `--help` parameter.

This is list of the most important parametes:

`--proceed` continue learning the latest model

`--dataset` directory of dataset

`--vocab_size` vocabulary size, count of most frequent words in dataset

`--epochs` num of epochs

`--hidden_size` hidden size of internal layers

`--logdir` log directory

`--model_name` this name uses for saving learning checkpoints

When training stops, the model state is saved to the `--model_name` file.

If `--proceed` is `True` script will try load last state of model (also training arguments of model), stored in `--model_name` and continue learning.

If `--proceed` is `False` script create new model and rewrite previous `--model_name` if exists.