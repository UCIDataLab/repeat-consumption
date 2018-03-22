"""
This file contains the main experiments done for the paper
"Predicting Consumption Patterns with Repeated and Novel Events"

It reproduces the main results from the paper for our method (tables 5 and 6).
reddit_top is a very large dataset, so it might take a long time to load and compute.
You can find the data used here: https://archive.ics.uci.edu/ml/datasets/Repeat+Consumption+Matrices
"""

from models.mixture_model import train_mixture_model
from util.data_io import get_dataset

for dataset in ['go_sf', 'go_ny', 'tw_oc', 'tw_ny', 'lastfm', 'reddit_sample', 'reddit_top']:
    print dataset
    train, val, test = get_dataset(dataset)
    train_mixture_model(train, val, test, dataset_name=dataset)
    train_mixture_model(train, val, test, dataset_name=dataset, method='recall')
    print


