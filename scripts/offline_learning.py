import os
import sys
import time
import random
import argparse
import numpy as np
import tensorflow as tf
from itertools import count
from tqdm import tqdm
ROOT_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
sys.path.insert(0, ROOT_PATH)
import lib

def trace(tensor):
    if len(tensor.shape) == 2:
        return sum([tensor[i,i] for i in range(tensor.shape[0])])
    if len(tensor.shape) == 3:
        return sum([tensor[:, i,i] for i in range(tensor.shape[1])])

def metrics(sigma, X, y):
    return np.mean(np.real((trace(X.dot(sigma)))-y)**2)

def train_model(true_state, args):
    dim = 2 ** args.n_qubits
    valid_proj = 10000
    valid_meas = 10000
    x_ph = tf.placeholder(dtype=tf.complex64, shape=[None, dim, dim])
    y_ph = tf.placeholder(dtype=tf.float32, shape=[None])
    model = lib.BaselineModel(x_ph, y_ph, learning_rate=args.lr)

    train_X, train_y = lib.generate_dataset(
        true_state, parsed_args.n_projectors, parsed_args.n_measurements)
    train_y = train_y.astype('float64')
    valid_X, valid_y = lib.generate_dataset(true_state, valid_proj, valid_meas)
    valid_y = valid_y.astype('float64')
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sigma = sess.run(model.sigma)
    start_loss = metrics(sigma, train_X, train_y)
    start_valid = metrics(sigma, valid_X, valid_y)
    tolerance = 10e-5 * start_valid

    last_loss = 0
    for t in tqdm(count()):
        indices = np.random.permutation(args.n_projectors)[:args.batch_size]
        loss_t, _ = sess.run([model.loss, model.optimize], {x_ph: train_X[indices], y_ph: train_y[indices]})

        if t % 100 == 0:
            sigma = sess.run(model.sigma)
            valid = metrics(sigma, valid_X, valid_y)
            if abs(last_loss - valid) < tolerance:
                break
            last_loss = valid

    final_loss = metrics(sigma, train_X, train_y)
    fidelity = lib.fidelity(true_state, sigma)
    return sigma, (fidelity, start_loss, final_loss, start_valid, valid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-qubits', type=int, required=True)
    parser.add_argument('--n-projectors', type=int, required=True)
    parser.add_argument('--n-measurements', type=int, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--exp-name', type=str, default='untitled')
    parser.add_argument('--seed', type=int, default=42)
    parsed_args = parser.parse_args()

    np.random.seed(parsed_args.seed)
    tf.set_random_seed(parsed_args.seed)
    random.seed(parsed_args.seed)

    t = time.time()
    true_state = lib.randomMixedState(2 ** parsed_args.n_qubits)
    pred_state, results = train_model(true_state, parsed_args)
    fidelity, start_loss, final_loss, start_valid, final_valid = results

    report = {
        'exp': parsed_args.exp_name,
        'qubits': parsed_args.n_qubits,
        'projectors': parsed_args.n_projectors,
        'measurements': parsed_args.n_measurements,
        'noise': parsed_args.noise,
        'fidelity': fidelity,
        'start_loss': start_loss,
        'final_loss': final_loss,
        'start_valid': start_valid,
        'final_valid': final_valid,
        'time': time.time() - t,
    }
    print(report)