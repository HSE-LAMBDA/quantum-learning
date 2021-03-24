from abc import ABC
from tqdm import tqdm, trange
import numpy as np
from typing import Optional
import sys
sys.path.append('../')
import lib


class AbstractAlgorithm(ABC):
    def __init__(self, n_qubits: int, rho: Optional[np.array], max_iters: int, patience: int):
        self.n_iters = 0
        self.best_score = 0
        self.best_score_iter = 0
        self.max_iters = max_iters
        self.n_qubits = n_qubits
        self.patience = patience
        self.rho = rho

    def reset(self):
        self.n_iters = 0
        self.best_score = 0
        self.best_score_iter = 0
        self.scores_history = []

    def fit(self, train_X, train_y):
        t = trange(self.max_iters, desc='Fidelity: 0', leave=True)

        for _ in t:
            self.fit_step_(train_X, train_y)
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

    def fit_step_(self, train_X, train_y):
        raise NotImplemented

    @property
    def rho_pred(self):
        raise NotImplemented
        
    def fidelity(self):
        try:
            return lib.fidelity(self.rho, self.rho_pred)
        except ValueError:
            return 0
    
    def score(self):
        return self.fidelity()
    
    def ttest(self, projectors, measurements_cnt):
        y_true = lib.generate_y(self.rho, projectors, measurements_cnt)
        y_pred = lib.generate_y(self.rho_pred, projectors, measurements_cnt)
        ttest = y_true - y_pred
        ttest /= np.sqrt(y_true*(1-y_true)*(2/measurements_cnt))
        return ttest

    def memory_consumption(self):
        raise NotImplemented

    def stop(self):
        return (self.n_iters - self.best_score_iter > self.patience) or (self.n_iters > self.max_iters)