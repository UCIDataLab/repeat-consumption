"""
This file contains various methods for learning the mixing weights, globally, per user etc.
It uses multi-processing to learn them, so the code can be lengthy at times.
The horsepower of this file is the method _learn_global_mixture_weights which implements the EM algorithm on the data.
Equation 4 from the paper Predicting Consumption Patterns with Repeated and Novel Events.
"""
import numpy as np
import time
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, coo_matrix
from util.array_util import data_to_sparse, get_max_row_column
from util import log_utils as log
from multiprocessing import cpu_count, Queue, Process
from scipy.stats import dirichlet


def get_train_global(train, val, test, is_eval=False):
    """
    Creates the self and global components. Self is a matrix of probability distributions (one for each user) and
    global is vector, that is the probability distribution of the entire population. Self is essentially the rows of
    the training data, normalized to sum to 1, after adding some smoothing.

    :param train: train data COO matrix
    :param val: validation data COO matrix
    :param test: test data COO matrix
    :param is_eval: Boolean, that defines if this data will be used for evaluation or training. The difference is that
    when evaluating, the validation data is added to the training, ow it is not.
    :return: self component, global component
    """
    r, c = get_max_row_column(train, val, test)
    smooth_factor = 1. / c
    if is_eval:
        train = np.vstack((train, val))
    train_matrix = csr_matrix(_make_train(train, smooth_factor))
    glb_matrix = csr_matrix(_make_global(train_matrix, smooth_factor))
    return train_matrix, glb_matrix


def _make_train(data, smooth_factor):
    """Returns normalized train data matrix after applying smoothing.

    Converts a coo matrix to sparse, adds a smoothing factor to users with no data, normalizes rows to sum to one.
    Args:
        data: Array of shape N x 3, that is a COO matrix (row, column, value).
        smooth_factor: A float to be added as a count to all rows that are all zero.
    Returns:
        sparse matrix, that is normalized by row.
    """
    train_matrix = data_to_sparse(data).tolil()
    user_counts = np.array(train_matrix.sum(axis=1))[:, 0]
    train_matrix[np.where(user_counts == 0)] = smooth_factor
    train_matrix = normalize(train_matrix, 'l1', axis=1)
    return train_matrix.tocsr()


def _make_global(train_matrix, smooth_factor):
    """Returns normalized vector that corresponds to the population preference.

    Sums the rows of a matrix, adds a smoothing factor to columns with no data, normalize vector to sum to one.
    Args:
        train_matrix: Sparse matrix that contains the prob distribution of each user.
        smooth_factor: A float to be added as a count to all columns that are zero.
    Returns:
        full vector, that is a probability distribution of all items.
    """
    glb_matrix = np.sum(train_matrix, axis=0)
    glb_matrix += smooth_factor
    glb_matrix /= np.sum(glb_matrix)
    return glb_matrix


def learn_global_mixture_weights(alpha, multinomials, val_mat, num_em_iter=100, tol=0.001):
    """
    Learning the mixing weights for mixture of k multinomials globally for all users. Each observation is a point in
    model.

    NOTE: In order for the algorithm to work, there can be no location that can get 0 probability by both the mem_mult
    and the mf_mult. In my runs, I use MPE to estimate the mf_mult while using MLE for the mum_mul. That way the mf_mult
    has no 0 values.

     INPUT:
    -------
        1. alpha:       <float / (2, ) ndarray>   Dirichlet prior for the pi learning. If <float> is given it is treated
                                                  as a flat prior. Has to be bigger than 1.
        2. multinomials: list[<(U, C) ndarray>]    each row is the multinomial parameter according to the "self" data
        3. mf_mult:     <(U, C) ndarray>    each row is the multinomial parameter according to the matrix factorization
        4. val_mat:     COO matrix    counts matrix to optimize on
        5. num_em_iter: <int>               number of em iterations
        6. tol:         <float>             convergence threshold

     OUTPUT:
    --------
        1. pis:  <(I, 2) ndarray>     each row is mixing weights for the i'th individual

     RAISE:
    -------
        1. ValueError:
                a. alphas are not bigger than 1
                b. the multinomial's rows don't sum to 1
                c. There is a location with both mults 0 (see NOTE)
    """
    log.info('Learning global mixing weights for all points')
    start = time.time()
    pi, _ = _learn_global_mixture_weights(alpha, multinomials, val_mat, num_em_iter, tol)
    total_time = time.time() - start
    log.info('Finished EM on all data (global mix weights). Total time: %d secs' % total_time)

    return pi


def learn_individual_mixing_weights(alpha, multinomials, val_mat, num_proc=None, num_em_iter=100, tol=0.001):
    """
    Learning the mixing weights for mixture of two multinomials. Each individual learns mixing weights.

    NOTE: In order for the algorithm to work, there can be no location that can get 0 probability by both the global
    and the user_memory_multinomial.

     INPUT:
    -------
        1. alpha:       <float / (2, ) ndarray>   Dirichlet prior for the pi learning. If <float> is given it is treated
                                                  as a flat prior. Has to be bigger than 1.
        2. multinomials: list[<(U, C) ndarray>]    each row is the multinomial parameter according to the "self" data
        3. val_mat:     <(U, C) ndarray>    counts matrix to optimize on
        4. num_proc:    <int> number of processes to be used.
        5. num_em_iter: <int>               number of em iterations
        6. tol:         <float>             convergence threshold

     OUTPUT:
    --------
        1. pis:  <(I, 2) ndarray>     each row is mixing weights for the i'th individual

     RAISE:
    -------
        1. ValueError:
                a. alphas are not bigger than 1
                b. the multinomial's rows don't sum to 1
                c. There is a location with both mults 0 (see NOTE)
    """
    n_users = multinomials[0].shape[0]

    assert len(alpha) == len(multinomials)
    if num_proc is None:
        num_proc = cpu_count()

    start = time.time()
    prior_strength = 1. * np.sum(val_mat) / val_mat.shape[0]  # average points

    pis, event_ll = _learn_individual_mixture_weights(n_users, alpha, multinomials, num_em_iter, tol, val_mat,
                                                      prior_strength, num_proc)
    total_time = time.time() - start
    log.info('User EM time: %d secs -- %.3f per user' % (total_time, total_time / n_users))
    mean = np.mean(np.array(pis), axis=0)
    log.info('Mean of mixing weights \t {self:%.4f\t population:%.4f}' % (mean[0], mean[1]))
    return np.array(pis)


def _learn_global_mixture_weights(alpha, multinomials, val_data, num_em_iter=100, tol=0.001):
    """
    Learning the mixing weights for mixture of two multinomials. Each observation is considered as a data point
    and the mixing weights (\pi) are learned using all the points.

    NOTE: In order for the algorithm to work, there can be no location that can get 0 probability by both the mem_mult
    and the mf_mult. In my runs, I use MPE to estimate the mf_mult while using MLE for the mum_mul. That way the mf_mult
    has no 0 values.


     INPUT:
    -------
        1. alpha:       <float / (2, ) ndarray>   Dirichlet prior for the pi learning. If <float> is given it is treated
                                                  as a flat prior. Has to be bigger than 1.
        2. multinomials:    list[<(U, C) ndarray>]    each row is the multinomial parameter according to the "self" data
        4. val_data:    <(N, 3) ndarray>    each row is [ind_id, loc_id, counts]
        5. num_em_iter: <int>               number of em iterations
        6. tol:         <float>             convergence threshold

     OUTPUT:
    --------
        1. pi:  <(N, ) ndarray>     mixing weights.
        2. log likelihood reached.

     RAISE:
    -------
        1. ValueError:
                a. alphas are not bigger than 1
                b. the multinomial's rows don't sum to 1
                c. _There is a location with both mults 0 (see NOTE)

    """
    num_comp = len(multinomials)
    if np.any(alpha <= 1):
        raise ValueError('alpha values have to be bigger than 1')

    for i, mult in enumerate(multinomials):
        if np.any(np.abs(np.sum(mult, axis=1) - 1) > 0.001):
            raise ValueError('component %d param is not a proper multinomial -- all rows must sum to 1' % i)

    if type(alpha) == float or type(alpha) == int:
        alpha = np.ones(num_comp) * alpha * 1.

    # Creating responsibility matrix and initializing it hard assignment on random
    log_like_tracker = [-np.inf]
    pi = np.ones(num_comp) / num_comp
    start = time.time()
    em_iter = 0
    for em_iter in xrange(1, num_em_iter + 1):
        # Evey 5 iteration we will compute the posterior log probability to see if we converged.
        if em_iter % 2 == 0:

            event_prob = _data_prob(pi, multinomials, val_data)
            event_prob = np.sum(event_prob, axis=0)  # prob

            # The data likelihood was computed for each location, but it should be in the power of the number
            # of observations there, or a product in the log space.
            data_likelihood = np.log(np.array(event_prob)) * val_data[:, 2]

            prior_probability = dirichlet.logpdf(pi, alpha=alpha)
            log_likelihood = np.sum(data_likelihood + prior_probability) / np.sum(val_data[:, 2])

            if np.abs(log_likelihood - log_like_tracker[-1]) < tol:
                log.debug('[iter %d] [Reached convergence.]' % em_iter)
                break

            log.debug('[iter %d] [Likelihood: [%.4f -> %.4f]]' % (em_iter, log_like_tracker[-1], log_likelihood))
            log_like_tracker.append(log_likelihood)

        # E-Step

        resp = _data_prob(pi, multinomials, val_data)

        if np.all(resp == 0):
            raise ValueError('0 mix probability')

        resp = np.array(resp).T
        resp = normalize(resp, 'l1', axis=1)

        resp = np.multiply(resp, val_data[:, 2][:, np.newaxis])
        pi = np.sum(resp, axis=0)
        pi += alpha - 1
        pi /= np.sum(pi)

    total_time = time.time() - start
    log.debug('Finished EM. Total time = %d secs -- %.3f per iteration' % (total_time, total_time / em_iter))

    data_log_like = _data_prob(pi, multinomials, val_data)
    data_log_like = np.sum(data_log_like, axis=0)
    ll = np.sum(np.log(np.array(data_log_like)) * val_data[:, 2]) / np.sum(val_data[:, 2])
    return pi, ll


def _learn_individual_mixture_weights(n_users, alpha, multinomials, max_iter, tol, val_mat, prior_strength, num_proc):
    """
    Learns the mixing weights for each individual user, uses multiple-processes to make it faster.

    :param n_users: Int, total number of users.
    :param alpha: prior (learned through global weights) for the pi's
    :param multinomials: List of components (Arrays of vectors).
    :param max_iter: max number of em iterations
    :param tol: convergence threshold
    :param val_mat: validation data to optimize on. U x C matrix.
    :param prior_strength: float, how much to increase the strength of the prior.
    :param num_proc: number of processes to be used.
    :return: 1. Matrix of mixing weights (Users x Components)
             2. Event log likelihood for validation data.
    """
    lls = np.ones(n_users)
    pis = np.tile(alpha, n_users).reshape(n_users, len(multinomials))
    pis = normalize(pis, 'l1', axis=1)  # pi's for each user.

    log.info('Doing individual weights with %d proc' % num_proc)
    mix_weights = []
    alpha *= prior_strength
    if any(alpha < 1):
        alpha += 1

    # multi-process. Essentially calls _mp_learn_user_mix for a set of users.
    batch_size = int(np.ceil(1. * n_users / num_proc))  # how many users per process
    args = (alpha, multinomials, val_mat, max_iter, tol)
    uids = range(n_users)
    queue = Queue()
    num_eof = 0
    proc_pool = []

    # set-up the processes
    for i in range(num_proc):
        p_uids = uids[i * batch_size:(i + 1) * batch_size]  # define which users this process will handle.
        if len(p_uids) == 0:
            break
        proc = Process(target=_mp_learn_user_mix, args=(queue, p_uids, args))
        proc_pool.append(proc)

    # start the processes
    [proc.start() for proc in proc_pool]

    # collect end tokens
    while num_eof < len(proc_pool):
        resp = queue.get()
        if type(resp) == str:
            num_eof += 1
        else:
            mix_weights.append(resp)
    [proc.join() for proc in proc_pool]
    queue.close()
    # end multi-process

    for id, u_mix_weights, u_ll in mix_weights:
        pis[id] = np.array(u_mix_weights)
        lls[id] = u_ll

    mask = np.where(lls != 1)

    lls = lls[mask] * np.squeeze(np.array(val_mat.sum(axis=1)))[mask]
    event_ll = np.sum(lls) / np.sum(val_mat)

    return pis, event_ll


def _data_prob(pi, multinomials, data):
    """
    Computes the probability of the data points for each component, given the mixing weights and the multinomials.

    :param pi: Mixing weights. (List of floats)
    :param multinomials: Component Multinomials, list of vectors or arrays. Arrays are user specific.
    :param data: data points, COO Matrix (N x 3).
    :return: list of prob vectors, one for each component. (matrix of components x events)
    """
    comp_prob = []
    for i, p_i in enumerate(pi):
        if multinomials[i].shape[0] == 1:
            mult = np.array(multinomials[i][0, data[:, 1].astype(int)].todense())
        else:
            mult = multinomials[i][data[:, 0].astype(int), data[:, 1].astype(int)]
        comp_prob.append(np.array(pi[i] * mult)[0])
    return comp_prob


def find(a):
    """Return the indices and values of the nonzero elements of a matrix

    Parameters
    ----------
    a : dense or sparse matrix
        Matrix whose nonzero elements are desired.

    Returns
    -------
    (I,J,V) : tuple of arrays
        I,J, and V contain the row indices, column indices, and values
        of the nonzero matrix entries.


    Examples
    --------
    >> from scipy.sparse import csr_matrix, find
    >> A = csr_matrix([[7.0, 8.0, 0],[0, 0, 9.0]])
    >> find(A)
    (array([0, 0, 1], dtype=int32), array([0, 1, 2], dtype=int32), array([ 7.,  8.,  9.]))

    """

    a = coo_matrix(a, copy=True)
    a.sum_duplicates()
    # remove explicit zeros
    nz_mask = a.data != 0
    return a.row[nz_mask], a.col[nz_mask], a.data[nz_mask]


def convert_sparse_to_coo(s_mat):
    """
    Converts a scipy.sparse (and even dense) matrix to a coo_matrix form where each row is [row, col, value].
    This is used to allow faster evaluation and optimization.

     INPUT:
    -------
        1. s_mat:       <(N, D) sparse_mat>     sparse/dense matrix or vector. It works with all!

     OUTPUT:
    --------
        1. coo_form:    <(nnz, 3) ndarray>      nnz is the number of non-zero elements.
                                                Each row is [row, col, val]. If the input is a vector all row values
                                                will be 0.
    """
    return np.vstack(find(s_mat)).T


def _mp_learn_user_mix(queue, ids, args):
    """Learns the mixture weights for each user, by calling learn global for only that user.

    After learning the weights, it puts the user_id, pi's and log likelihood in the queue. After completing all users
    it puts $ in the queue to signify end.

    :param queue: Queue to write and communicate with main process.
    :param ids: List of user ids this process is responsible for.
    :param args: arguments that will be used to call _learn_global_mixture_weights.
    :return: -
    """
    alpha, multinomials, val_mat, num_em_iter, tol = args
    for i in ids:
        # The way the global em is implemented, allows me to simply call it with the i_val_data and it will only
        # compute the \pi as a function of that user.
        i_val_data = convert_sparse_to_coo(val_mat[i])
        if len(i_val_data) > 0:

            # The learning method treats the multinomials as matrices. So I have to wrap it in an array.
            # All the rows in i_val_data are going to be 0 because I'm converting a single row_vector.
            user_multinomials = []
            for m in multinomials:
                if m.shape[0] == 1:
                    user_multinomials.append(m)
                else:
                    user_multinomials.append(m[i])
            i_pi, i_ll = _learn_global_mixture_weights(alpha, user_multinomials, i_val_data, num_em_iter, tol)

            queue.put((i, i_pi, i_ll))

    queue.put('$')
