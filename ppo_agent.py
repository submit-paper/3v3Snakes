import inspect
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.train import AdamOptimizer


class PPOAgent():
    def __init__(self, model_cls, observation_space, action_space, config=None,
                 gamma=0.99, lam=0.95, lr=2.5e-4, clip_range=0.1, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 epochs=4, nminibatches=4, *args, **kwargs):

        # Define parameters
        self.gamma = gamma
        self.lam = lam
        self.base_lr = self.lr = lr
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.nminibatches = nminibatches

        # Default model config
        if config is None:
            config = {'model': [{'model_id': 'policy_model'}]}

        # Model related objects
        self.model = None
        self.sess = None
        self.train_op = None
        self.pg_loss = None
        self.vf_loss = None
        self.entropy = None
        self.clip_rate = None
        self.kl = None

        # Placeholder for training targets
        self.advantage_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.return_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.old_neglogp_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.old_v_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[])
        
        self.model_cls = model_cls
        self.observation_space = observation_space
        self.action_space = action_space

        if config is not None:
            self.load_config(config)

        self.model_instances = None
        self._init_model_instances(config)

        self.build()

    def build(self) -> None:
        self.model = self.model_instances[-1]

        self.entropy = tf.reduce_mean(self.model.entropy)

        vpredclipped = self.old_v_ph + tf.clip_by_value(self.model.vf - self.old_v_ph, -self.clip_range,
                                                        self.clip_range)
        # Unclipped value
        vf_losses1 = tf.square(self.model.vf - self.return_ph)

        # Clipped value
        vf_losses2 = tf.square(vpredclipped - self.return_ph)
        self.vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(self.old_neglogp_ph - self.model.neglogp_a)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -self.advantage_ph * ratio
        pg_losses2 = -self.advantage_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

        # Final PG loss
        self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        # Total loss
        loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

        # Stat
        self.kl = tf.reduce_mean(self.model.neglogp_a - self.old_neglogp_ph)
        clipped = tf.logical_or(ratio > (1 + self.clip_range), ratio < (1 - self.clip_range))
        self.clip_rate = tf.reduce_mean(tf.cast(clipped, tf.float32))

        params = tf.trainable_variables(self.model.scope)
        trainer = tf.train.AdamOptimizer(learning_rate=self.lr_ph, epsilon=1e-5)
        grads_and_var = trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads_and_var = list(zip(grads, var))

        self.train_op = trainer.apply_gradients(grads_and_var)
        self.sess = self.model.sess
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

    def sample(self, state: Any, *args, **kwargs) -> Tuple[Any, dict]:
        action, value, neglogp = self.model.forward(state)
        return action, {'value': value, 'neglogp': neglogp}

    def learn(self, training_data, *args, **kwargs):
        data = [training_data[key] for key in ['state', 'return', 'action', 'value', 'neglogp', 'legal_action']]
        nbatch = len(data[0])
        nbatch_train = nbatch // self.nminibatches

        inds = np.arange(nbatch)
        stats = defaultdict(list)
        for _ in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in data)
                ret = self.train(*slices)

                for k, v in ret.items():
                    stats[k].append(v)

        return {k: np.array(v).mean() for k, v in stats.items()}

    def train(self, obs, returns, actions, values, neglogps, legal_action):
        advs = returns - values
        # advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.model.x_ph: obs,
            self.model.a_ph: actions,
            self.advantage_ph: advs,
            self.return_ph: returns,
            self.lr_ph: self.lr,
            self.old_neglogp_ph: neglogps,
            self.old_v_ph: values,
            self.model.legal_action: legal_action
        }
        _, pg_loss, vf_loss, entropy, clip_rate, kl = self.sess.run(
            [self.train_op, self.pg_loss, self.vf_loss, self.entropy, self.clip_rate, self.kl], td_map)
        return {
            'pg_loss': pg_loss,
            'vf_loss': vf_loss,
            'entropy': entropy,
            'clip_rate': clip_rate,
            'kl': kl
        }

    def prepare_training_data(self, trajectory: List[Tuple[Any, Any, Any, Any, Any, dict]]) -> Dict[str, np.ndarray]:
        mb_states, mb_actions, mb_rewards, mb_dones, next_state, mb_extras = trajectory
        mb_values = np.asarray([extra_data['value'] for extra_data in mb_extras])
        mb_neglogp = np.asarray([extra_data['neglogp'] for extra_data in mb_extras])
        mb_legalac = np.asarray([extra_data['legal_action'] for extra_data in mb_extras])

        last_values = self.model.forward(next_state)[1]
        mb_values = np.concatenate([mb_values, last_values[np.newaxis]])

        mb_deltas = mb_rewards + self.gamma * mb_values[1:] * (1.0 - mb_dones) - mb_values[:-1]

        nsteps = len(mb_states)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(nsteps)):
            nextnonterminal = 1.0 - mb_dones[t]
            mb_advs[t] = lastgaelam = mb_deltas[t] + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values[:-1]
        data = [sf01(arr) for arr in [mb_states, mb_returns, mb_actions, mb_values[:-1], mb_neglogp, mb_legalac]]
        name = ['state', 'return', 'action', 'value', 'neglogp', 'legal_action']
        return dict(zip(name, data))

    def post_process_training_data(self, training_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return training_data

    def set_weights(self, weights, *args, **kwargs) -> None:
        self.model.set_weights(weights)

    def get_weights(self, *args, **kwargs) -> Any:
        return self.model.get_weights()

    def save(self, path: Path, *args, **kwargs) -> None:
        self.model.save(path)

    def load(self, path: Path, *args, **kwargs) -> None:
        self.model.load(path)

    def preprocess(self, state: Any, *args, **kwargs) -> Any:
        pass

    def update_sampling(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        pass

    def update_training(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        pass

    def _init_model_instances(self, config: Union[dict, None]) -> None:
        """Initialize model instances"""
        self.model_instances = []

        def create_model_instance(_c: dict):
            ret = {}
            for k, v in _c.items():
                if k in valid_config:
                    ret[k] = v
                else:
                    warnings.warn(f"Invalid config item '{k}' ignored", RuntimeWarning)
            self.model_instances.append(self.model_cls(self.observation_space, self.action_space, **ret))

        if config is not None and 'model' in config:
            model_config = config['model']
            valid_config = get_config_params(self.model_cls)

            if isinstance(model_config, list):
                for _, c in enumerate(model_config):
                    create_model_instance(c)
            elif isinstance(model_config, dict):
                create_model_instance(model_config)
        else:
            self.model_instances.append(self.model_cls(self.observation_space, self.action_space))


def get_config_params(obj_or_cls) -> List[str]:
    """
    Return configurable parameters in 'Agent.__init__' and 'Model.__init__' which appear after 'config'
    :param obj_or_cls: An instance of 'Agent' / 'Model' OR their corresponding classes (NOT base classes)
    :return: A list of configurable parameters
    """
    sig = list(inspect.signature(obj_or_cls.__init__).parameters.keys())

    config_params = []
    config_part = False
    for param in sig:
        if param == 'config':
            # Following parameters should be what we want
            config_part = True
        elif param in {'args', 'kwargs'}:
            pass
        elif config_part:
            config_params.append(param)

    return config_params

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
