#   __author__ = 'dimitrios'
from __future__ import division
import numpy as np
from numba import jit
from util.array_util import data_to_sparse


def evaluate(all_multinomials, test_data, method, k=100):
    """
    Evaluates the model (resulted multinomials) on the test data, based on the method selected.
    Prints evaluation metric (averaged per event).

    :param all_multinomials:
    :param test_data: COO matrix with test data.
    :param method: string, can be 'logp' for log probability or 'recall' for recall@k
    :param k: The k from recall@k. If method is logP, then this does nothing.
    :return: -
    """
    test_points = np.repeat(test_data[:, :-1], test_data[:, -1].astype(int), axis=0).astype(int)
    test_probs = all_multinomials[list(test_points.T)]
    if method.lower() == 'logp':
        return np.mean(np.log(test_probs))  # per event
    elif method.lower() == 'recall':
        return recall_at_top_k(data_to_sparse(test_data), all_multinomials, k)
    else:
        print 'I do not know this evaluation method'


@jit(nogil=True)
def rates_to_exp_order(rates, argsort, exp_order, M):
    prev_score = 0
    prev_idx = 0
    prev_val = rates[argsort[0]]
    for i in range(1, M):
        if prev_val == rates[argsort[i]]:
            continue

        tmp = 0
        for j in range(prev_idx, i):
            exp_order[argsort[j]] = prev_score + 1
            tmp += 1

        prev_score += tmp
        prev_val = rates[argsort[i]]
        prev_idx = i

    # For the last equalities
    for j in range(prev_idx, i + 1):
        exp_order[argsort[j]] = prev_score + 1


@jit(nogil=True)
def rates_mat_to_exp_order(rates, argsort, exp_order, N, M):
    for i in range(N):
        rates_to_exp_order(rates[i], argsort[i], exp_order[i], M)


# @jit(cache=True)
@jit
def fix_exp_order(rates, exp_order, k, N):
    for i in range(N):
        mask = np.where(exp_order[i] <= k)[0]
        if len(mask) <= k:
            exp_order[i] = 0
            exp_order[i, mask] = 1
        else:
            max_val = np.max(exp_order[i, mask])
            max_val_mask = np.where(exp_order[i] == max_val)[0]
            exp_order[i] = 0
            exp_order[i, mask] = 1
            exp_order[i, max_val_mask] = (k - max_val + 1) / max_val_mask.shape[0]


@jit
def recall_at_top_k(test_counts, scores, k):
    """
    Args
        1. test_counts: sparse matrix of the test data (could be non sparse too)
        2. scores:      scores, probability, mf -- whatever that has the score for each item
                        where higher is better
        3. k:           the top k.
    """
    argsort = np.argsort(-scores, axis=1)
    exp_order = np.zeros(scores.shape)

    rates_mat_to_exp_order(scores, argsort, exp_order, scores.shape[0], scores.shape[1])
    fix_exp_order(scores, exp_order, k, scores.shape[0])
    recall_in = test_counts.multiply(exp_order)
    u_recall = recall_in.sum(axis=1) / test_counts.sum(axis=1)  # gives runtime warning, but its ok. NaN are handled.
    return np.mean(u_recall[~np.isnan(u_recall)])  # nan's do not count.
