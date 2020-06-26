import numpy as np
import pandas as pd
import linecache
from scipy.linalg import sqrtm
from .simulator import projectorOnto, measure
 

def generate_dataset(target_state, projectors_cnt, measurements_cnt, noise=0, shuffle=True):
    dim = target_state.shape[0]
    train_X = []
    projectors = []  # E
    for _ in range(projectors_cnt):
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        proj = projectorOnto(psi)
        proj /= np.trace(proj)
        projectors.append(proj)

        if noise != 0:
            psi = psi + np.random.normal(0, noise)
            proj = projectorOnto(psi)
            proj /= np.trace(proj)
        train_X.append(proj)

    train_y = []
    for proj in projectors:
        ones, zeroes = measure(measurements_cnt, target_state, proj)
        train_y.append(ones / measurements_cnt)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    indices = np.arange(train_X.shape[0])
    if shuffle: np.random.shuffle(indices)
    return train_X[indices], train_y[indices]


def fidelity(state_1, state_2):
    state_1_sqrt = sqrtm(state_1)
    F = np.dot(np.dot(state_1_sqrt, state_2), state_1_sqrt)
    return (np.trace(sqrtm(F)) ** 2).real


def read_data(path):
    target_state_string = linecache.getline(path, 3)[16:-2]
    target_state_raw = np.array(target_state_string.split(), dtype=np.float32)
    target_state = projectorOnto(target_state_raw[::2] + 1j * target_state_raw[1::2])

    df = pd.read_csv(path, skiprows=4, delim_whitespace=True, header=None)
    total_ticks = df[0].values[-1]
    train_y = df[1].values * df.shape[0] / (total_ticks*target_state.shape[0])

    psi_list = df.iloc[:, 2:].values[:, ::2] + 1j * df.iloc[:, 2:].values[:, 1::2]
    train_X = []
    for psi in psi_list:
        proj = projectorOnto(psi)
        train_X.append(proj)
    return target_state, np.array(train_X), train_y
