import os
import random
import tensorflow as tf
import numpy as np
import torch


def seed_everything(seed=1):
    seed_value = seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    tf.random.set_random_seed(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)

