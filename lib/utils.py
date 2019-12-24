import numpy as np
from scipy.linalg import sqrtm
from .simulator import randomPureState, measure


def generate_dataset(target_state, projectors_cnt, measurements_cnt, shuffle=True):
    dim = target_state.shape[0]
    projectors = []  # E
    for _ in range(projectors_cnt):
        proj = randomPureState(dim)  # E_m
        projectors.append(proj)
    train_X = []
    train_y = []
    for proj in projectors:
        measurements = measure(measurements_cnt, target_state, proj)
        train_X.append(proj)
        train_y.append(measurements[0] / measurements.sum())
        # for _ in range(measurements[0]):
        #     train_X.append(proj)
        #     train_y.append(1)
        # for _ in range(measurements[1]):
        #     train_X.append(proj)
        #     train_y.append(0)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    indices = np.arange(train_X.shape[0])
    if shuffle: np.random.shuffle(indices)
    return train_X[indices], train_y[indices]


def fidelity(state_1, state_2):
    state_1_sqrt = sqrtm(state_1)
    F = np.dot(np.dot(state_1_sqrt, state_2), state_1_sqrt)
    return (np.trace(sqrtm(F)) ** 2).real
