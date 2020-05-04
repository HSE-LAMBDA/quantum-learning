import numpy as np
from scipy.linalg import sqrtm
from .simulator import projectorOnto, measure
 
import tntorch as tn
import torch
import numpy as np
import time
from functools import reduce


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

def optimize(tensors, loss_function, eval_function, optimizer=torch.optim.Adam, tol=1e-4, max_iter=1e4, print_freq=500, verbose=True, patience=10):
    """
    High-level wrapper for iterative learning.
    Default stopping criterion: either the absolute (or relative) loss improvement must fall below `tol`.
    In addition, the rate loss improvement must be slowing down.
    :param tensors: one or several tensors; will be fed to `loss_function` and optimized in place
    :param loss_function: must take `tensors` and return a scalar (or tuple thereof)
    :param optimizer: one from https://pytorch.org/docs/stable/optim.html. Default is torch.optim.Adam
    :param tol: stopping criterion
    :param max_iter: default is 1e4
    :param print_freq: progress will be printed every this many iterations
    :param verbose:
    """

    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    parameters = []
    for t in tensors:
        if isinstance(t, tn.Tensor):
            parameters.extend([c for c in t.cores if c.requires_grad])
            parameters.extend([U for U in t.Us if U is not None and U.requires_grad])
        elif t.requires_grad:
            parameters.append(t)
    if len(parameters) == 0:
        raise ValueError("There are no parameters to optimize. Did you forget a requires_grad=True somewhere?")

    optimizer = optimizer(parameters)
    losses = []
    converged = False
    start = time.time()
    iter = 0
    best_iter = 0
    best_eval_loss = eval_function(*tensors)
    while True:
        optimizer.zero_grad()
        loss = loss_function(*tensors)
        if not isinstance(loss, (tuple, list)):
            loss = [loss]
        losses.append(reduce(lambda x, y: x + y, loss))
        if len(losses) >= 2:
            delta_loss = (losses[-1] - losses[-2])
        else:
            delta_loss = float('-inf')
        if iter >= 2 and ((tol is not None and (losses[-1] <= tol or -delta_loss / losses[-1] <= tol) and losses[-2] - \
                losses[-1] < losses[-3] - losses[-2]) or (iter - best_iter > patience)):
            converged = True
            break
        if iter == max_iter:
            break
        eval_loss = eval_function(*tensors)
        if verbose and iter % print_freq == 0:
            print(f'iter: {iter} |  loss: {loss[0].item()} | eval_loss: {eval_loss} | total time: {time.time() - start}')
        losses[-1].backward(retain_graph=True)
        optimizer.step()
        iter += 1
        if eval_loss < best_eval_loss:
            best_iter = iter
            best_eval_loss = eval_loss
    if verbose:
        print(f'iter: {iter} |  loss: {loss[0].item()} | eval_loss: {eval_loss} | total time: {time.time() - start}')
        if converged:
            print(' <- converged (tol={})'.format(tol))
        else:
            print(' <- max_iter was reached: {}'.format(max_iter))