import os
import sys
import time
import random
import argparse
import torch
import tntorch as tn
import numpy as np
ROOT_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
sys.path.insert(0, ROOT_PATH)
import lib

def trace(tensor):
    if len(tensor.shape) == 2:
        return sum([tensor[i,i] for i in range(tensor.shape[0])])
    if len(tensor.shape) == 3:
        return sum([tensor[:, i,i] for i in range(tensor.shape[1])])

def loss(sigma_real, sigma_imag, X, y):
    res = 0
    for E_m,y_m in zip(X, y.astype('float64')):
        E_real, E_imag = [tn.Tensor(x) for x in [np.real(E_m), np.imag(E_m)]]
        res += ((E_real.dot(sigma_real)+E_imag.dot(sigma_imag)-y_m)**2)
    return res / len(X)

def metrics(sigma, X, y):
    return np.mean(np.real((trace(X.dot(sigma)))-y)**2)

def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, tn.Tensor):
        x = x.torch() 
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x

     
def train_model(true_state, parsed_args):
    dim = true_state.shape[0]
    valid_proj = 10000
    valid_meas = 10000
    train_X, train_y = lib.generate_dataset(
        true_state, parsed_args.n_projectors, parsed_args.n_measurements)
    train_y = train_y.astype('float64')
    valid_X, valid_y = lib.generate_dataset(true_state, valid_proj, valid_meas)
    valid_y = valid_y.astype('float64')
    
    sigma = lib.simulator.randomMixedState(dim)
    sigma_real, sigma_imag = [tn.Tensor(x, ranks_tt=parsed_args.tensor_rank, requires_grad=True) 
        for x in [np.real(sigma), np.imag(sigma)]]
        
    start_loss = metrics(sigma, train_X, train_y)
    start_valid = metrics(sigma, valid_X, valid_y)
    tolerance = 10e-2 * start_loss 
    tn.optimize(
        [sigma_real, sigma_imag], 
        lambda x, y: loss(x, y, train_X, train_y), 
        verbose=False,
        tol=tolerance)
    sigma = check_numpy(sigma_real) + 1j * check_numpy(sigma_imag)
    
    final_loss = metrics(sigma, train_X, train_y)
    final_valid = metrics(sigma, valid_X, valid_y)
    return sigma, (start_loss, final_loss, start_valid, final_valid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-qubits', type=int, required=True)
    parser.add_argument('--n-projectors', type=int, required=True)
    parser.add_argument('--n-measurements', type=int, required=True)
    parser.add_argument('--tensor-rank', type=int, required=True)
    parser.add_argument('--exp-name', type=str, default='untitled')
    parser.add_argument('--seed', type=int, default=42)
    parsed_args = parser.parse_args()

    np.random.seed(parsed_args.seed)
    torch.manual_seed(parsed_args.seed)
    random.seed(parsed_args.seed)

    t = time.time()
    true_state = lib.randomMixedState(2 ** parsed_args.n_qubits)
    pred_state, results = train_model(true_state, parsed_args)
    start_loss, final_loss, start_valid, final_valid = results

    
    report = {
        'exp': parsed_args.exp_name,
        'qubits': parsed_args.n_qubits,
        'projectors': parsed_args.n_projectors,
        'measurements': parsed_args.n_measurements,
        'tensor_rank': parsed_args.tensor_rank,
        'start_loss': start_loss,
        'final_loss': final_loss,
        'start_valid': start_valid,
        'final_valid': final_valid,
        'time': time.time() - t,
    }
    print(report)
    
