import argparse
import sys
sys.path.append('../')
import lib
import numpy as np
from meg import MEG, DDMEG
from lptn import LPTN
from cholesky import Cholesky
from tqdm import trange
import time
import gc
import json
import os

RESULTS_DIR = 'results/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for quantum learning')
    parser.add_argument('executions_cnt', type=int,
                        help='Number of executions to run')
    parser.add_argument('algorithm', type=str, choices=['meg', 'dd_meg', 'lptn', 'cholesky'])
    parser.add_argument('--n_qubits', type=int,
                        help='Number of qubits')
    parser.add_argument('--data_type', type=str, choices=['pure', 'mixed', 'real'], default='real')
    parser.add_argument('--projectors_cnt', type=int, default=1000)
    parser.add_argument('--measurements_cnt', type=int, default=1000)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=2000)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--file', type=str, help='Used only in with real data')
    parser.add_argument('--tensor_rank', type=int, help='Tensor rank for LPTN algorithm')
    args = parser.parse_args()

    kwargs = {
        'max_iters': args.max_iters,
        'patience': args.patience
    }
    if args.algorithm == 'lptn': kwargs['tensor_rank']= args.tensor_rank
    if args.eta: kwargs['eta'] = args.eta # learning rate

    def load_data():
        if args.data_type == 'real':
            rho, train_X, train_y = lib.read_data(args.file)
            dim = rho.shape[0]
            n_qubits = int(np.log2(dim))
        else:
            n_qubits = args.n_qubits
            dim = 2 ** n_qubits
            rho = {'pure': lib.randomPureState, 'mixed': lib.randomMixedState}[args.data_type](dim)
            train_X, train_y = lib.generate_dataset(rho, args.projectors_cnt, args.measurements_cnt)
        train_y = train_y.astype('float64')
        return rho, n_qubits, train_X, train_y


    rho, n_qubits, train_X, train_y = load_data()

    gpu_mem_free = lambda: sum(lib.utils.get_gpu_memory_free())

    results = []
    gpu_memory_before = gpu_mem_free()
    ram_memory_used_before = lib.utils.get_ram_memory_usage()
    algorithm = {'meg': MEG,
                 'dd_meg': DDMEG,
                 'lptn': LPTN,
                 'cholesky': Cholesky
                 }[args.algorithm](n_qubits, rho, **kwargs)
    exp_id = len(os.listdir(RESULTS_DIR))

    for _ in trange(args.executions_cnt):
        start_ts = time.time()
        algorithm.fit(train_X, train_y)
        results.append({'Fidelity': algorithm.score(), 'time': time.time() - start_ts,
                        'num_steps': algorithm.n_iters,
                        'gpu_mem': gpu_memory_before - gpu_mem_free(),
                        'ram_mem': lib.utils.get_ram_memory_usage() - ram_memory_used_before})
        algorithm.reset()
        del rho, train_X, train_y
        gc.collect()
        rho, n_qubits, train_X, train_y = load_data()

    with open(f'{RESULTS_DIR}/{exp_id}.json', 'w') as f:
        f.write(json.dumps({
            'results': results,
            'args': vars(args)
        }))