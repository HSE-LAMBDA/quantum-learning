from abstract_algorithm import AbstractAlgorithm
import sys
sys.path.append('../')
import lib
import torch
import numpy as np
from typing import Optional

class Cholesky(AbstractAlgorithm):
    def __init__(self, n_qubits: int, rho: Optional[np.array], max_iters: int,
                 patience: int, eta: float = 1e-3,
                 device: str = 'auto'):
        super(Cholesky, self).__init__(n_qubits, rho, max_iters, patience)
        self.eta = eta
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reset()

    def reset(self):
        self.M = lib.ComplexTensor(lib.simulator.randomMixedState(self.rho.shape[0]))
        self.M.requires_grad = True
        self.opt = lib.optim.Ranger([self.M], lr=self.eta)


    def fit_step_(self, train_X, train_y):
        sigma = self.M.t(conjugate=True).mm(self.M)
        norm = torch.trace(sigma.real)
        def btrace(tensor):
            return torch.einsum('bii->b', tensor)
        E_m = lib.ComplexTensor(train_X)
        product_real, _ = E_m.bmm(sigma)
        loss = ((btrace(product_real) / norm - torch.from_numpy(train_y.astype('float64')))**2).mean()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.n_iters += 1

    def score(self):
        sigma = self.M.t(conjugate=True).mm(self.M)
        norm = torch.trace(sigma.real)
        sigma = sigma / norm
        dim = 2**self.n_qubits
        sigma = sigma.detach().numpy()[:dim] + 1j * sigma.detach().numpy()[dim:]
        return lib.fidelity(self.rho, sigma)