import tensorflow as tf
import numpy as np


def fix_tf_seed(seed=0):
    tf.random.set_seed(seed)


def fix_np_seed(seed=0):
    np.random.seed(seed)
