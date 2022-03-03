from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import get_session

'''
这里卷积的输入必须都写成NHWC的格式,也就是channel必须放最后(因为原代码不完善),
然而tensorflow也有直接支持此格式的卷积层,故没什么大问题.
'''
# from https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding
# original name: periodic_padding_flexible
def circular_pad(tensor, axis,padding=1):# 实现循环padding
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    """
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding,int):
        padding = (padding,)

    ndim = len(tensor.shape)
    for ax,p in zip(axis,padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left

        ind_right = [slice(-p,None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right,middle,left], axis=ax)

    return tensor

def cirbasicblock(input,title,filter_num,firstpad,stride=1):
    activ = tf.nn.relu
    convout = activ(conv(circular_pad(input,(1,2),firstpad), '{}c1'.format(title), nf=filter_num, rf=3, stride=stride, init_scale=np.sqrt(2)))
    convout = conv(circular_pad(convout,(1,2),(1,1)), '{}c2'.format(title), nf=filter_num, rf=3, stride=1, init_scale=np.sqrt(2))
    if stride != 1:
        resout = conv(input, '{}c_res'.format(title), nf=filter_num, rf=1, stride=stride, init_scale=np.sqrt(2))
    else:
        resout = input

    output = activ(convout + resout)
    return output    


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init
    
def placeholder(dtype=tf.float32, shape=None):
    return tf.placeholder(dtype=dtype, shape=combined_shape(None, shape))
    

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    '''
    nf: 几个卷积核(输出的channel数)
    rf: 卷积核宽
    '''
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            b = tf.reshape(b, bshape)
        return tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format) + b


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w) + b


def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x


class ACCNNModel():
    def __init__(self, observation_space, action_space, config=None, model_id='0', *args, **kwargs):
        with tf.variable_scope(model_id):
            self.x_ph = placeholder(shape=observation_space, dtype='uint8')
            self.encoded_x_ph = tf.to_float(self.x_ph)
            self.a_ph = placeholder(dtype=tf.int32)
            self.legal_action = placeholder(shape = (None,))

        self.logits = None
        self.vf = None

        session = get_session()
        self.sess = session
        self.observation_space = observation_space
        self.action_space = action_space
        self.model_id = model_id
        self.config = config
        self.scope = model_id

        # Build up model
        self.build()

        # Build assignment ops
        self._weight_ph = None
        self._to_assign = None
        self._nodes = None
        self._build_assign()

        # Build saver
        self.saver = tf.train.Saver(tf.trainable_variables())

        pd = CategoricalPd(self.logits)
        self.action = pd.sample()
        self.neglogp = pd.neglogp(self.action)
        self.neglogp_a = pd.neglogp(self.a_ph)
        self.entropy = pd.entropy()

        init_op = tf.global_variables_initializer()
        session.run(init_op)

    def set_weights(self, weights, *args, **kwargs) -> None:
        feed_dict = {self._weight_ph[var.name]: weight
                     for (var, weight) in zip(tf.trainable_variables(scope=self.scope), weights)}

        self.sess.run(self._nodes, feed_dict=feed_dict)

    def get_weights(self, *args, **kwargs) -> Any:
        return self.sess.run(tf.trainable_variables(self.scope))

    def save(self, path: Path, *args, **kwargs) -> None:
        self.saver.save(self.sess, str(path))

    def load(self, path: Path, *args, **kwargs) -> None:
        self.saver.restore(self.sess, str(path))

    def _build_assign(self):
        self._weight_ph, self._to_assign = dict(), dict()
        variables = tf.trainable_variables(self.scope)
        for var in variables:
            self._weight_ph[var.name] = tf.placeholder(var.value().dtype, var.get_shape().as_list())
            self._to_assign[var.name] = var.assign(self._weight_ph[var.name])
        self._nodes = list(self._to_assign.values())

    def forward(self, states: Any, legal_action, *args, **kwargs) -> Any:
        return self.sess.run([self.action, self.vf, self.neglogp], feed_dict={self.x_ph: states, self.legal_action: legal_action})

    def build(self, *args, **kwargs) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('cnn_base'):
                # scaled_images = tf.cast(self.encoded_x_ph, tf.float32) / 255.
                input_images = tf.cast(self.encoded_x_ph, tf.float32)
                activ = tf.nn.relu

                outstem = activ(conv(circular_pad(input_images,(1,2),(1,1)), 'c_stem', nf=16, rf=3, stride=1, init_scale=np.sqrt(2)))
                outstem = tf.nn.max_pool(circular_pad(outstem,(1,2),(1,1)),[1,2,2,1],[1,1,1,1],padding='VALID')
                
                outresnet = cirbasicblock(outstem,"rb1_1",16,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb1_2",16,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb2_1",16,(1,1),2)
                outresnet = cirbasicblock(outresnet,"rb2_2",16,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb3_1",32,(1,1),2)
                outresnet = cirbasicblock(outresnet,"rb3_2",32,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb4_1",32,(0,1),2)
                outresnet = cirbasicblock(outresnet,"rb4_2",32,(1,1),1)
                outresnet = conv_to_fc(outresnet)

                latent = activ(fc(outresnet, 'fc1', nh=64, init_scale=np.sqrt(2)))

            with tf.variable_scope('pi'):
                pih1 = activ(fc(latent, 'pi_fc1', 64, init_scale=0.01))
                pih2 = activ(fc(pih1, 'pi_fc2', 64, init_scale=0.01))
                logits_without_mask = fc(pih2, 'pi_fc3', self.action_space, init_scale=0.01)
                self.logits = logits_without_mask + 1000. *tf.to_float(self.legal_action-1)

            with tf.variable_scope('v'):
                vh1 = activ(fc(latent, 'v_fc1', 64, init_scale=0.01))
                vh2 = activ(fc(vh1, 'v_fc2', 64, init_scale=0.01))
                self.vf = tf.squeeze(fc(vh2, 'v_fc3', 1, init_scale=0.01), axis=1)

class CategoricalPd:
    def __init__(self, logits):
        self.logits = logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def logp(self, x):
        return -self.neglogp(x)

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

            x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        else:
            # already encoded
            assert x.shape.as_list() == self.logits.shape.as_list()

        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=x)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
