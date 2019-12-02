import unittest

import numpy as np
import tensorflow as tf

from tf2rl.misc.fix_seed import fix_tf_seed, fix_np_seed


class TestFixSeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seed = 12345
        cls.shape = (10,)

    def test_fix_tf_seed(self):
        fix_tf_seed(self.seed)
        expected = tf.random.uniform(shape=self.shape)
        fix_tf_seed(self.seed)
        results = tf.random.uniform(shape=self.shape)

    def test_fix_np_seed(self):
        fix_np_seed(self.seed)
        expected = np.random.uniform(size=self.shape)
        fix_np_seed(self.seed)
        results = np.random.uniform(size=self.shape)


if __name__ == "__main__":
    unittest.main()
