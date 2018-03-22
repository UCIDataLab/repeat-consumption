"""
General utility methods that are used for array manipulation. Conversion from COO -> CSR and vice versa.
"""
import numpy as np
from scipy.sparse import coo_matrix


def sparse_to_data_array(matrix, dtype=np.float32, maintain_size=True):
    """Converts a sparse matrix into a COO data array (numpy array of shape (n, 3)"""
    matrix.eliminate_zeros()
    data = np.zeros((matrix.nnz, 3), dtype=dtype)
    data[:, 0] = matrix.nonzero()[0]
    data[:, 1] = matrix.nonzero()[1]
    data[:, 2] = matrix.data
    if maintain_size:
        m, n = matrix.shape
        data = np.vstack((data, [m - 1, n - 1, 0]))  # insert a 0 value to maintain the size
    return data


def data_to_sparse(data, shape=None, csc=False):
    """Takes in (n,3) array of data and returns csr matrix for that"""
    if csc:
        data_matrix = coo_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=shape).tocsc()
    else:
        data_matrix = coo_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=shape).tocsr()

    data_matrix.eliminate_zeros()
    return data_matrix


def get_max_row_column(train, val, test):
    """Returns the total number of rows, columns from train, validation and test arrays in COO form."""
    r = max(max(train[:, 0]), max(val[:, 0]), max(test[:, 0])).astype(int) + 1
    c = max(max(train[:, 1]), max(val[:, 1]), max(test[:, 1])).astype(int) + 1
    return r, c
