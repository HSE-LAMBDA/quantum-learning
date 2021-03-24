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
MAX_EXECUTIONS_CNT = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for quantum learning')
    parser.add_argument('experiment_id', type=int, help='Experiment id')
    parser.add_argument('executions_cnt', type=int,
                        help='Number of executions to run')
    parser.add_argument('algorithm', type=str, choices=['meg', 'dd_meg', 'lptn', 'cholesky'])
    parser.add_argument('--n_qubits', type=int,
                        help='Number of qubits')
    parser.add_argument('--data_type', type=str, choices=['pure', 'mixed', 'real'], default='real')
    parser.add_argument('--projectors_cnt', type=int, default=10000)
    parser.add_argument('--measurements_cnt', type=int, default=1000)
    parser.add_argument('--max_iters', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--file', type=str, help='Used only in with real data')
    parser.add_argument('--tensor_rank', type=int, help='Tensor rank for LPTN algorithm')
    parser.add_argument('--noise', type=float, default=0., help='Noise in projectors')
    parser.add_argument('--test_size', type=int, default=10000, help='Number of projectors for test')
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
            
            _, train_X, train_y = lib.generate_dataset(rho, args.projectors_cnt, args.measurements_cnt, args.noise)
        _, test_X, test_y = lib.generate_dataset(rho, args.test_size, args.measurements_cnt, args.noise)
        return rho, n_qubits, train_X, train_y, test_X, test_y


    rho, n_qubits, train_X, train_y, test_X, test_y = load_data()

    results = []

    executions_cnt = min(args.executions_cnt, MAX_EXECUTIONS_CNT)
    for _ in trange(executions_cnt):
        algorithm = {'meg': MEG,
             'dd_meg': DDMEG,
             'lptn': LPTN,
             'cholesky': Cholesky
             }[args.algorithm](n_qubits, rho, **kwargs)

        start_ts = time.time()
        algorithm.fit(train_X, train_y)
        results.append({'Fidelity': algorithm.fidelity(), 
                        'ttest_train': algorithm.ttest(train_X, args.measurements_cnt).squeeze().tolist(),
                        'ttest_test': algorithm.ttest(test_X, args.measurements_cnt).squeeze().tolist(),
                        'time': time.time() - start_ts,
                        'fidelities': algorithm.scores_history,
                        'num_steps': algorithm.n_iters,
                        'gpu_mem': lib.utils.get_gpu_memory_usage(),
                        'ram_mem': lib.utils.get_ram_memory_usage()})
        del rho, train_X, train_y, test_X, test_y, algorithm
        gc.collect()
        rho, n_qubits, train_X, train_y, test_X, test_y = load_data()

    with open(f'{RESULTS_DIR}/{args.experiment_id}.json', 'w') as f:
        f.write(json.dumps({
            'results': results,
            'args': vars(args)
        }))