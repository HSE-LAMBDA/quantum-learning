import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from abstract_algorithm import AbstractAlgorithm
import sys
import numpy as np
import lib
from tqdm import trange
from typing import Optional

class MEG(AbstractAlgorithm):
    """
    MEG from 1802.09025
    """
    def __init__(self,
                 n_qubits: int,
                 rho: Optional[np.array],
                 max_iters: int = 10000,
                 patience: int = 400,
                 eta: float = 0.1):
        super(MEG, self).__init__(n_qubits, rho, max_iters, patience)
        self.eta = eta
        self.reset()

    def reset(self):
        super().reset()
        self.sigma = tf.Variable(tf.eye(2 ** self.n_qubits, dtype=tf.complex128) * (2 ** -self.n_qubits))

    def fit(self, train_X, train_y):
        train_X, train_y = [tf.Variable(x, dtype=tf.complex128) for x in [train_X, train_y]]
        super(MEG, self).fit(train_X, train_y)

    def fit_step_(self, train_X, train_y):
        self.n_iters += 1

        trace = tf.linalg.trace(tf.matmul(train_X, self.sigma))
        grad = 2 * tf.einsum('i,ijk->jk', trace - train_y, train_X)

        # MEGUPDATE
        G = tf.math.log(self.sigma) - self.eta * grad
        self.sigma = tf.exp(G)
        self.sigma = tf.Variable(self.sigma / tf.linalg.trace(self.sigma))
        return self.sigma

    @property
    def rho_pred(self):
        return self.sigma.numpy()

    def memory_consumption(self):
        return self.sigma.size()

class DDMEG(MEG):
    """
    Adaptive MEG with doubling trick, arXiv:2006.01013
    """
    def fit(self, train_X, train_y):
        x_var = tf.Variable(train_X, dtype=tf.complex128)
        y_var = tf.Variable(train_y, dtype=tf.complex128)

        self.update_eta()
        losses = []

        t = trange(self.max_iters, desc='Fidelity: 0', leave=True)
        for _ in t:
            self.fit_step_(x_var, y_var)

            trace = tf.linalg.trace(tf.matmul(x_var, self.sigma))
            loss = (y_var - trace) ** 2
            losses.append(np.mean(loss.numpy()))

            if np.real(sum(losses)) >= 2 ** (self.n_iters + 1):
                # Update learning rate
                self.update_eta()
                losses = []

            score = self.score()
            self.scores_history.append(score)
            if score > self.best_score:
                self.best_score_iter = self.n_iters
                self.best_score = score
            if self.stop():
                return
            t.set_description("Fidelity: %.2f" % score)
            t.refresh()  # to show immediately the update
        return self.scores_history

    def reset(self):
        super().reset()
        self.update_eta()

    def update_eta(self):
        beta = self.n_iters + 1
        self.eta = min(np.sqrt(self.n_qubits*np.log(2)/(2**beta+1)), 0.5)