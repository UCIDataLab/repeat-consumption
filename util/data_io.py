from util.io import load_txt
from util.paths import data_dir
from util.array_util import data_to_sparse
from util import log_utils as log

import os

data_url = 'https://archive.ics.uci.edu/ml/datasets/Repeat+Consumption+Matrices'


def get_dataset(dataset_name, return_csr=False):
    """
    Returns train, val and test data for a dataset. They are in COO form, as required by the mixture_model methods.

    :param dataset_name: name of directory where the train, val and test files are
    :param return_csr: Boolean that defines whether the arrays should be COO (default) of shape N x 3, or CSR (UxC)
    :return: three COO matrices, corresponding to train, val and test
    """

    train_name = os.path.join(data_dir, dataset_name, 'train.csv')
    val_name = os.path.join(data_dir, dataset_name, 'validation.csv')
    test_name = os.path.join(data_dir, dataset_name, 'test.csv')
    log.info('Loading data for %s' % dataset_name)
    try:
        train = load_txt(train_name, verbose=False)
        val = load_txt(val_name, verbose=False)
        test = load_txt(test_name, verbose=False)

        if return_csr:
            train = data_to_sparse(train)
            val = data_to_sparse(val)
            test = data_to_sparse(test)

        return train, val, test
    except IOError as e:

        print 'There was a problem loading the data for dataset %s.\n The correct structure is to have ' % dataset_name,
        print 'a ./data directory with the dataset name (%s), and inside it there need to be a ' % dataset_name,
        print 'train.csv, validation.csv and test.csv files. Eg path: %s' % train_name
        print 'Each file must contain a COO numpy matrix.'

        print 'The data used in the paper, can be found in %s' % data_url
        print 'Instructions on how to use them etc.'
        print e.message
