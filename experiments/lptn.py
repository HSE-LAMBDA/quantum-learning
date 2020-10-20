from abstract_algorithm import AbstractAlgorithm
import tntorch as tn
import torch
import numpy as np
from typing import Optional
import sys
sys.path.append('../')
import lib

class LPTN(AbstractAlgorithm):
    """
    Locally purified tensor networks
    """
    def __init__(self, n_qubits: int, rho: Optional[np.array], max_iters: int,
                 patience: int, eta: float, tensor_rank: Optional[int] = None,
                 batch_size: int = 100,
                 device: str = 'auto'):
        super().__init__(n_qubits, rho, max_iters, patience)
        self.eta = eta
        if tensor_rank is None:
            tensor_rank = 2**n_qubits
        self.tensor_rank = tensor_rank
        self.batch_size = batch_size
        if device == 'auto': self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else: self.device = torch.device(device)
        self.reset()

    def reset(self):
        super(LPTN, self).reset()
        sigma = lib.randomMixedState(2 ** self.n_qubits)
        self.sigma_real, self.sigma_imag = [tn.Tensor(x, ranks_tt=self.tensor_rank, requires_grad=True, device=self.device) for x in
                                            [np.real(sigma), np.imag(sigma)]]

    @staticmethod
    def cholesky(sigma_real, sigma_imag):
        sigma_real, sigma_imag = sigma_real.dot(sigma_real, k=1) + sigma_imag.dot(sigma_imag, k=1), sigma_real.dot(
            sigma_imag, k=1) - sigma_imag.dot(sigma_real, k=1)
        trace = sum([sigma_real[i, i] for i in range(sigma_real.shape[0])])
        sigma_real, sigma_imag = [x / trace for x in [sigma_real, sigma_imag]]
        return sigma_real, sigma_imag

    @staticmethod
    def trace(tensor):
        if len(tensor.shape) == 2:
            return sum([tensor[i, i] for i in range(tensor.shape[0])])
        if len(tensor.shape) == 3:
            return sum([tensor[:, i, i] for i in range(tensor.shape[1])])

    def fit(self, train_X, train_y):
        def loss(sigma_real, sigma_imag):
            sigma_real, sigma_imag = LPTN.cholesky(sigma_real, sigma_imag)
            res = 0
            idx = np.random.choice(np.arange(train_X.shape[0]), self.batch_size)
            for E_m, y_m in zip(train_X[idx], train_y[idx].astype('float64')):
                E_real, E_imag = [tn.Tensor(x, device=self.device) for x in [np.real(E_m), np.imag(E_m)]]
                res += ((E_real.dot(sigma_real) + E_imag.dot(sigma_imag) - y_m) ** 2)
            #     return res/(initial_trace*train_X.shape[0])
            return res

        def eval_loss(sigma_real, sigma_imag):  # any score function can be used here
            sigma_real_, sigma_imag_ = LPTN.cholesky(sigma_real, sigma_imag)
            sigma = sigma_real_.torch().detach().cpu().numpy() + 1j * sigma_imag_.torch().detach().cpu().numpy()
            return -lib.fidelity(sigma, self.rho)

        self.n_iters, self.best_score, self.best_score_iter = lib.tn_optimize([self.sigma_real, self.sigma_imag], loss, eval_loss,
                        tol=0, patience=self.patience, print_freq=10, lr=self.eta, max_iter=self.max_iters)

    def score(self):
        sigma_real_, sigma_imag_ = LPTN.cholesky(self.sigma_real, self.sigma_imag)
        sigma = sigma_real_.torch().detach().cpu().numpy() + 1j * sigma_imag_.torch().detach().cpu().numpy()
        return lib.utils.fidelity(sigma, self.rho)