from abstract_algorithm import AbstractAlgorithm
import tntorch as tn
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import gc
from typing import Optional
import sys
sys.path.append('../')
import lib

class LPTN(AbstractAlgorithm):
    """
    Locally purified tensor networks
    """
    def __init__(self, n_qubits: int, rho: Optional[np.array], max_iters: int,
                 patience: int, eta: float=1e-3, tensor_rank: Optional[int] = None,
                 batch_size: int = None,
                 device: str = 'auto'):
        super().__init__(n_qubits, rho, max_iters, patience)
        self.eta = eta
        if tensor_rank is None:
            tensor_rank = 2**n_qubits
        self.tensor_rank = tensor_rank

        if batch_size is None: self.batch_size = int(10000/max(1, 2**(n_qubits-6)))
        else: self.batch_size = batch_size

        if device == 'auto': self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else: self.device = torch.device(device)
        self.reset()

    def reset(self):
        super(LPTN, self).reset()
        sigma = lib.randomMixedState(2 ** self.n_qubits)
        self.sigma_real, self.sigma_imag = [tn.Tensor(x, ranks_tt=self.tensor_rank, requires_grad=True, device=self.device) for x in
                                            [np.real(sigma), np.imag(sigma)]]
        parameters = []
        for t in [self.sigma_real, self.sigma_imag]:
            if isinstance(t, tn.Tensor):
                parameters.extend([c for c in t.cores if c.requires_grad])
                parameters.extend([U for U in t.Us if U is not None and U.requires_grad])
            elif t.requires_grad:
                parameters.append(t)
        if len(parameters) == 0:
            raise ValueError("There are no parameters to optimize. Did you forget a requires_grad=True somewhere?")

        self.opt = lib.optim.Ranger(parameters, lr=self.eta)

    @staticmethod
    def trace(tensor):
        if len(tensor.shape) == 2:
            return sum([tensor[i, i] for i in range(tensor.shape[0])])
        if len(tensor.shape) == 3:
            return sum([tensor[:, i, i] for i in range(tensor.shape[1])])

    @staticmethod
    def cholesky(sigma_real, sigma_imag):
        sigma_real, sigma_imag = sigma_real.dot(sigma_real, k=1) + sigma_imag.dot(sigma_imag, k=1), sigma_real.dot(
            sigma_imag, k=1) - sigma_imag.dot(sigma_real, k=1)
        trace = LPTN.trace(sigma_real)
        sigma_real, sigma_imag = [x / trace for x in [sigma_real, sigma_imag]]
        return sigma_real, sigma_imag

    def fit_step_(self, train_X, train_y):
        self.n_iters += 1
        if self.batch_size < train_X.shape[0]:
            dataset = TensorDataset(*[torch.from_numpy(x.astype('float64')) for x in [np.real(train_X), np.imag(train_X), train_y]])
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X_real, batch_X_imag, batch_y in dataloader:
                E_real, E_imag = [tn.Tensor(x.permute((1,2,0)), device=self.device) for x in [batch_X_real, batch_X_imag]]
                batch_y = tn.Tensor(batch_y, device=self.device)
                self.opt.zero_grad()
                loss = tn.metrics.dist((E_real.dot(self.sigma_real, k=2) + E_imag.dot(self.sigma_imag, k=2)), batch_y)
                loss.backward()
                self.opt.step()
                del E_real, E_imag, batch_X_real, batch_X_imag, batch_y
                gc.collect()
                torch.cuda.empty_cache()
        else:
            batch_y = tn.Tensor(torch.from_numpy(train_y.astype('float64')), device=self.device)
            E_real, E_imag = [tn.Tensor(torch.from_numpy(x.astype('float64')).permute((1,2,0)), device=self.device) for x in [np.real(train_X), np.imag(train_X)]]
            self.opt.zero_grad()
            loss = tn.metrics.dist((E_real.dot(self.sigma_real, k=2) + E_imag.dot(self.sigma_imag, k=2)), batch_y)
            loss.backward()
            self.opt.step()
            del E_real, E_imag, batch_y
            gc.collect()
            torch.cuda.empty_cache()


    @property
    def rho_pred(self):
        sigma_real_, sigma_imag_ = LPTN.cholesky(self.sigma_real, self.sigma_imag)
        return sigma_real_.torch().detach().cpu().numpy() + 1j * sigma_imag_.torch().detach().cpu().numpy()