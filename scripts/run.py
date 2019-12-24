import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
ROOT_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
sys.path.insert(0, ROOT_PATH)
import lib


def init_writer(args):
    current_time = '_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}:{:0>2d}'.format(*time.gmtime()[:6])
    log_dir = os.path.join(ROOT_PATH, './logs', args.exp_name + current_time)
    writer = SummaryWriter(log_dir, comment=args.exp_name)
    writer.add_text('params', ','.join(map(str, [args.exp_name, args.n_qubits, args.n_proj,
                                                 args.n_measur, args.lr, args.seed])))
    return writer


def train_model(rho, args):
    dim = 2 ** args.n_qubits
    x_ph = tf.placeholder(dtype=tf.complex64, shape=[None, dim, dim])
    y_ph = tf.placeholder(dtype=tf.float32, shape=[None])
    model = lib.BaselineModel(x_ph, y_ph, learning_rate=args.lr)

    X_train, y_train = lib.generate_dataset(rho, args.n_proj, args.n_measur)
    y_train = y_train.astype('float64')
    measurements_rho = np.array([lib.simulator.bornRule(x, rho) for x in X_train])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    writer = init_writer(args)
    fidelity_history = []

    for t in tqdm(count()):
        indices = np.random.permutation(args.n_proj)[:args.batch_size]
        loss_t, _ = sess.run([model.loss, model.optimize], {x_ph: X_train[indices], y_ph: y_train[indices]})
        writer.add_scalar('train/loss', loss_t, t)

        if t % 25 == 0:
            sigma = sess.run(model.sigma)
            measurements_sigma = np.array([lib.simulator.bornRule(x, sigma) for x in X_train])

            diff = abs(measurements_sigma - measurements_rho)
            fidelity = lib.fidelity(rho, sigma)
            writer.add_scalar('valid/metrics', diff.mean(), t)
            writer.add_scalar('valid/fidelity', fidelity, t)

            fidelity_history.append(fidelity)
            if len(fidelity_history) - np.argmax(fidelity_history) > 50:
                break

    return sess.run(model.sigma), np.max(fidelity_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-qubits', type=int, required=True)
    parser.add_argument('--n-proj', type=int, required=True)
    parser.add_argument('--n-measur', type=int, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--exp-name', type=str, default='untitled')
    parser.add_argument('--seed', type=int, default=42)
    parsed_args = parser.parse_args()

    np.random.seed(parsed_args.seed)
    tf.set_random_seed(parsed_args.seed)

    t = time.time()
    true_state = lib.randomMixedState(2 ** parsed_args.n_qubits)
    pred_state, fidelity = train_model(true_state, parsed_args)

    results_file = open('./logs/results', 'a', buffering=1)
    report = {
        'exp': parsed_args.exp_name,
        'qubits': parsed_args.n_qubits,
        'proj': parsed_args.n_proj,
        'meas': parsed_args.n_measur,
        'fidel': fidelity,
        'time': time.time() - t,
    }
    results_file.write(str(report) + '\n')
    results_file.close()
