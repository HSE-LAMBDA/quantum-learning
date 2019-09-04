import os
import sys
import time
import numpy as np
import tensorflow as tf
import lib

param_set = {}
for i, param in enumerate(sys.argv[1:]):
    if i % 2 == 0:
        if param == 'exp_path':
            param_set[param] = sys.argv[i + 2]
        else:
            param_set[param] = int(sys.argv[i + 2])
print(param_set)

t0 = time.time()

rho = np.array([[1., 0.], [0., 0.]])  # target state
dim = rho.shape[0]

projectors_cnt = param_set['projectors']
measurements_cnt = param_set['measurements']

train_size = projectors_cnt * measurements_cnt

x_ph = tf.placeholder(dtype=tf.complex64, shape=[None, dim, dim])
y_ph = tf.placeholder(dtype=tf.float32, shape=[None])
model = lib.BaselineModel(x_ph, y_ph)

loss_history = []
metrics_history = []
fidelity_history = []
sess = tf.InteractiveSession()

for i in range(10):

    train_X, train_y = lib.generate_dataset(rho, projectors_cnt, measurements_cnt)
    train_y = train_y.astype('float64')
    sess.run(tf.global_variables_initializer())
    measurements_rho = np.array([lib.simulator.bornRule(x, rho) for x in train_X])

    epochs = int(1e5)
    last_loss = int(1e9)
    loss_t = 0

    for t in range(epochs):
        loss_t, _ = sess.run([model.loss, model.optimize], {x_ph: train_X, y_ph: train_y})
        if abs(last_loss - loss_t) < 1e-7:
            break
        last_loss = loss_t

    loss_history.append(loss_t)

    sigma = sess.run(model.sigma)
    measurements_sigma = np.array([lib.simulator.bornRule(x, sigma) for x in train_X])

    diff = abs(measurements_sigma - measurements_rho)
    metrics_history.append(diff.mean())
    fidelity_history.append(lib.fidelity(rho, sigma))

t1 = time.time()
print('Time: ', t1 - t0)

results_file = open(os.path.join(param_set['exp_path'], 'results'), 'a', buffering=1)
param_set['loss'] = np.round(np.mean(loss_history), 5)
param_set['metric'] = np.round(np.mean(metrics_history), 5)
param_set['fidelity'] = np.round(np.mean(fidelity_history), 5)
del param_set['exp_path']
results_file.write(str(param_set) + '\n')
results_file.close()
