import time
import sys
sys.path.insert(0, '..')

import simulator
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

param_set = {}
for i, param in enumerate(sys.argv[1:]):
    if i % 2 == 0:
        if param == 'exp_name':
            param_set[param] = sys.argv[i + 2]
        else:
            param_set[param] = int(sys.argv[i + 2])
print(param_set)

def tf_fidelity(state_1, state_2):
    state_1_sqrt = tf.linalg.sqrtm(state_1)
    F = tf.matmul(tf.matmul(state_1_sqrt, state_2), state_1_sqrt)
    return tf.real(tf.trace(tf.linalg.sqrtm(F)) ** 2)

t0 = time.time()

rho = np.array([[1., 0.], [0., 0.]]) # target state
dim = rho.shape[0]

projectors_cnt = param_set['projectors']
measurements_cnt = param_set['measurements']

train_size = projectors_cnt * measurements_cnt

sess = tf.InteractiveSession()

M_real, M_img = tf.Variable(tf.random_normal([dim, dim])), tf.Variable(tf.random_normal([dim, dim]))
M = tf.complex(M_real, M_img)

rho_ph = tf.placeholder(dtype=M.dtype, shape=[dim, dim])
x_ph = tf.placeholder(dtype=M.dtype, shape=[None, dim, dim])
y_ph = tf.placeholder(dtype=tf.float32, shape=[None])

sigma = tf.matmul(tf.conj(M), M, transpose_a=True)
sigma /= tf.trace(sigma)

measurements = tf.real(tf.trace(tf.einsum('bij,jk->bik', x_ph, rho_ph)))
prediction = tf.real(tf.trace(tf.einsum('bij,jk->bik', x_ph, sigma)))
loss = tf.losses.mean_squared_error(prediction, y_ph)

update_M = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss, var_list=[M_real, M_img])
sess.run(tf.global_variables_initializer())


loss_history = []
mean_metrics_history = []
max_metrics_history = []
fidelity_history = []

for i in range(10):
    
    train_X, train_y = simulator.generate_dataset(rho, projectors_cnt, measurements_cnt)
    train_y = train_y.astype('float64')
    sess.run(tf.global_variables_initializer())
    measurements_rho = sess.run(measurements, {x_ph: train_X, rho_ph: rho})
    
    epoches = int(1e5)
    last_loss = int(1e9)
    
    for t in range(epoches):
        loss_t, _ = sess.run([loss, update_M], {x_ph: train_X, y_ph: train_y})
        if abs(last_loss - loss_t) < 1e-7:
            break
        last_loss = loss_t
        
    loss_history.append(loss_t)
    
    

    sigma_pred = sess.run(sigma)
    measurements_sigma = sess.run(measurements, {x_ph: train_X, rho_ph: sigma_pred})
    
    diff = abs(measurements_sigma - measurements_rho)
    max_metrics_history.append(diff.max())
    mean_metrics_history.append(diff.mean())
    
    fidelity = tf_fidelity(rho.astype(np.complex64), sigma_pred)
    fidelity_history.append(sess.run(fidelity))
    
t1 = time.time()
print('Time: ', t1 - t0)
        
fout = open(param_set['exp_name'] + '.results', 'a', buffering=1)
param_set['loss'] = round(np.mean(loss_history), 5)
param_set['mean'] = round(np.mean(mean_metrics_history), 5)
param_set['max'] = round(np.mean(max_metrics_history), 5)
param_set['fidelity'] = round(np.mean(fidelity_history), 5)
del param_set['exp_name']
fout.write(str(param_set) + '\n')
fout.close()