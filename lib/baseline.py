import tensorflow as tf


class BaselineModel:
    def __init__(self, data, target, learning_rate=1e-3):
        dim = int(data.get_shape()[-1])
        M_real, M_img = tf.Variable(tf.random.normal([dim, dim])), tf.Variable(tf.random.normal([dim, dim]))
        M = tf.complex(M_real, M_img)

        sigma = tf.matmul(tf.math.conj(M), M, transpose_a=True)
        self._sigma = sigma / tf.linalg.trace(sigma)

        prediction = tf.math.real(tf.linalg.trace(tf.einsum('bij,jk->bik', data, self._sigma)))
        self._loss = tf.compat.v1.losses.mean_squared_error(prediction, target)
        self._optimize = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=learning_rate).minimize(self._loss, var_list=[M_real, M_img])

    @property
    def sigma(self):
        return self._sigma

    @property
    def optimize(self):
        return self._optimize

    @property
    def loss(self):
        return self._loss
