import logging
import operator
import functools

import torch
import numpy as np
from sklearn import datasets
from sklearn import preprocessing as prep


def get_logger(name: str = __file__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(levelname)-8s] %(asctime)s [%(filename)s] [%(funcName)s:%(lineno)d] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    return logger


def gather_nd(params, indices):
    """
    the input indices must be a 2d tensor in the form of [[a,b,..,c],...],
    which represents the location of the elements.
    """
    max_value = functools.reduce(operator.mul, list(params.size())) - 1
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    idx[idx < 0] = 0
    idx[idx > max_value] = 0
    return torch.take(params, idx)


def convert_svmlight_to_torch(path, dtype=np.float32):
    data, _ = datasets.load_svmlight_file(path, zero_based=True, dtype=dtype)
    coo = data.tocoo(copy=False)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


def prep_standardize(x):
    """
    takes sparse input and compute standardized version
    Note:
        cap at 5 std
    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled
