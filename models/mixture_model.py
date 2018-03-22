"""
This file contains a method that implements the model from the paper 'Predicting Consumption Patterns with Repeated and
Novel Events.'
The method self_global implements the mixture model described there, that identifies personalized mixing-weights,
between two components: The self history (exploit) and the global preferences (explore).
"""
import os
import numpy as np

from models.evaluation import evaluate
from models.mm_functions import get_train_global, learn_individual_mixing_weights, learn_global_mixture_weights
from util.io import load_pickle, save_pickle
from util.array_util import data_to_sparse

from util.paths import results_dir  # go here to define paths


def train_mixture_model(train, val, test, method='logP', recall_k=100, dataset_name='d_name', overwrite=False,
                        num_proc=None):
    """
    Runs the main experiment of the paper, finding the best mixing weights per user for these two components.
    Learns the weights per user and saves them in a file. If file exists it just loads it.
    It evaluates on the test set.

    There is a memory component, where a person has been in the past (exploit), and global component, which is the
    population preferences (explore).

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array Users x Categories. N is the number of entries in the array.

    :param train: train data COO matrix
    :param val: validation data COO matrix
    :param test: test data COO matrix
    :param method: Method of evaluation. Can be 'logP' or 'recall' for log probability per event, or recall@k
    :param recall_k: the k for recall@k. If method is 'logP' this does nothing.
    :param dataset_name: Name of the directory the results will be saved.
    :param overwrite: Boolean, on whether to overwrite learned weights or read them if they exist.
    :param num_proc: Number of processes to be used. If none, all the processors in the machine will be used.

    :return: returns an array of mixing weights, which is n_users x 2 (2 components, self and global)
    """

    filename = os.path.join(results_dir, 'mixture_model', dataset_name, 'mixing_weights.pkl')
    if os.path.exists(filename) and not overwrite:
        mix_weights = load_pickle(filename, False)
    else:
        train_matrix, global_matrix = get_train_global(train, val, test)
        components = [train_matrix, global_matrix]  # can add more components here

        mix_weights = learn_mixing_weights(components, val, num_proc=num_proc)
        save_pickle(filename, mix_weights, False)

    evaluate_method(train, val, test, mix_weights, method, recall_k)
    return mix_weights


def learn_mixing_weights(components, validation_data, num_proc=None):
    """Runs the Smoothed Mixture model on the number of components. Each component is an array of U x C
    (user x categories). This runs in parallel for efficiency.

    :param components: list of matrices (CSR or full) -- all must have the same size
    :param validation_data: COO matrix of validation data
    :param num_proc: number of processes to be used.
    :return the mixing weights for each user (or the entire population).
    """

    alpha = 1.001  # very small prior for global.
    global_mix_weights = learn_global_mixture_weights(alpha, components, validation_data)  # learn global mix weights.

    val_data = data_to_sparse(validation_data)

    # use global mixing weights as prior for individual ones.
    user_mix_weights = learn_individual_mixing_weights(global_mix_weights, components, val_data, num_proc)

    return user_mix_weights


def evaluate_method(train, val, test, mix_weights, method='logP', recall_k=100):
    """
    Evaluates the mixing weights for a given method (logP or recall).

    :param train: train data COO matrix
    :param val: validation data COO matrix
    :param test: test data COO matrix
    :param mix_weights: the mixing weights for each user (or the entire population).
    :param method: string, 'logP' or 'recall'
    :param recall_k: if method is recall, then this is the k for recall@k
    :return: evaluation metric averaged per event.
    """
    eval_train_matrix, eval_glb_matrix = get_train_global(train, val, test, is_eval=True)
    eval_components = [eval_train_matrix, eval_glb_matrix]

    user_multinomials = mix_multinomials(eval_components, mix_weights)
    per_event = evaluate(user_multinomials, test, method, k=recall_k)

    print '%s: %.4f' % (method, per_event)
    return per_event


def mix_multinomials(components, mixing_weights):
    """Returns the multinomial distribution for each user after mixing them.

    :param components: List of components. arrays or vectors.
    :param mixing_weights: List of mixing weights (for each component). If mixing weights is an array, then it means
    there is one mixing weight per user. Otherwise, they are global.
    :return: array of probability distribution for each user
    """
    if mixing_weights.ndim == 2:
        return _user_individual_multinomial(components, mixing_weights)
    else:
        return _user_global_multinomials(components, mixing_weights)


def _user_global_multinomials(components, mixing_weights):
    """The mixing weights are the same for each user.

    :param components: List of components
    :param mixing_weights: List of mixing weights (one for each component)
    :return: array of probability distribution for each user
    """

    result = np.zeros(components[0].shape)
    for i, c in enumerate(components):
        result += mixing_weights[i] * c
    return np.array(result)


def _user_individual_multinomial(components, mixing_weights):
    """ The mixing weights are different for each user.

    :param components: List of components
    :param mixing_weights: List of mixing weights (for each component, and each user)
    :return: array of probability distribution for each user
    """
    result = np.zeros(components[0].shape)
    for i, c in enumerate(components):
        c = np.array(c.todense())
        result += mixing_weights[:, i][:, np.newaxis] * c
    return result
